import os
import json
from datetime import datetime
import numpy as np

from biomechanics_config import FOLDER_PACJENTOW, NAZWY_KATOW_3D, KOMPENSACJE_DEFINICJE


def analizuj_pelna_sekwencje(historia_katow, historia_kompensacji):
    if not historia_katow or not historia_kompensacji:
        return None

    arr_katy = np.array(historia_katow, dtype=float)
    arr_komp = np.array(historia_kompensacji, dtype=float)

    statystyki_stawow = []
    for idx, nazwa in enumerate(NAZWY_KATOW_3D):
        kolumna = arr_katy[:, idx]
        k_min = float(np.percentile(kolumna, 5))
        k_max = float(np.percentile(kolumna, 95))
        statystyki_stawow.append({
            "staw": nazwa,
            "min": round(k_min, 1),
            "max": round(k_max, 1),
            "rom": round(k_max - k_min, 1)
        })

    szczyty_komp = np.percentile(arr_komp, 90, axis=0)
    srednie_komp = np.mean(arr_komp, axis=0)

    wykryte_kompensacje = []
    wszystkie_kompensacje = []
    suma_kar = 0.0

    for idx, definicja in enumerate(KOMPENSACJE_DEFINICJE):
        nazwa = definicja["nazwa"]
        prog = definicja["prog"]
        szczyt = float(szczyty_komp[idx])
        srednia = float(srednie_komp[idx])
        przekroczenie = max(0.0, szczyt - prog)

        wpis = {
            "nazwa": nazwa,
            "szczyt": round(szczyt, 1),
            "srednia": round(srednia, 1),
            "prog": prog,
            "przekroczenie": round(przekroczenie, 1),
            "status": "KOMPENSACJA" if przekroczenie > 0.5 else "NORMA"
        }
        wszystkie_kompensacje.append(wpis)

        if przekroczenie > 0.5:
            wykryte_kompensacje.append(wpis)
            suma_kar += przekroczenie * 1.8

    wykryte_kompensacje.sort(key=lambda x: x["przekroczenie"], reverse=True)
    mqi = round(max(0.0, min(100.0, 100.0 - suma_kar)), 1)

    return {
        "mqi": mqi,
        "liczba_klatek": len(historia_katow),
        "stawy_rom": statystyki_stawow,
        "wykryte_kompensacje": wykryte_kompensacje,
        "wszystkie_kompensacje": wszystkie_kompensacje
    }


def utworz_rekord_raportu(id_pacjenta, nazwa_cwiczenia, zrodlo, wyniki_analizy, sciezka_wideo_hud=""):
    bezpieczny_id = "".join(c for c in id_pacjenta if c.isalnum() or c in ("_", "-")).strip() or "Pacjent_01"
    return {
        "pacjent": bezpieczny_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cwiczenie": nazwa_cwiczenia,
        "zrodlo": zrodlo,
        "wideo_z_szkieletem": sciezka_wideo_hud,
        "mqi": wyniki_analizy["mqi"],
        "liczba_klatek": wyniki_analizy["liczba_klatek"],
        "wykryte_kompensacje": wyniki_analizy["wykryte_kompensacje"],
        "wszystkie_kompensacje": wyniki_analizy["wszystkie_kompensacje"],
        "stawy_rom": wyniki_analizy["stawy_rom"]
    }


def zapisz_wynik_pacjenta(rekord):
    bezpieczny_id = rekord["pacjent"]
    folder = os.path.join(FOLDER_PACJENTOW, bezpieczny_id)
    os.makedirs(folder, exist_ok=True)
    znacznik = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sciezka = os.path.join(folder, f"{znacznik}_{rekord['cwiczenie']}.json")
    with open(sciezka, "w", encoding="utf-8") as f:
        json.dump(rekord, f, indent=2, ensure_ascii=False)
    return sciezka


def eksportuj_raport_tekstowy(rekord, sciezka_docelowa):
    """Zapisuje czytelny raport diagnostyczny dla pacjenta / fizjoterapeuty w pliku .txt lub .html."""
    if sciezka_docelowa.endswith(".html"):
        wiersze_komp = "".join(
            f"<tr style='background:{'#fee2e2' if k['status']=='KOMPENSACJA' else '#f0fdf4'}'>"
            f"<td>{k['nazwa']}</td><td>{k['szczyt']}°</td><td>≤ {k['prog']}°</td>"
            f"<td><b>{k['status']}</b></td></tr>"
            for k in rekord["wszystkie_kompensacje"]
        )
        wiersze_rom = "".join(
            f"<tr><td>{s['staw']}</td><td>{s['min']}°</td><td>{s['max']}°</td><td><b>{s['rom']}°</b></td></tr>"
            for s in rekord["stawy_rom"]
        )
        html = f"""<!DOCTYPE html>
<html lang="pl">
<head><meta charset="utf-8"><title>Raport KineTrack 3D - {rekord['pacjent']}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 30px; color: #1e293b; }}
h1, h2 {{ color: #0f172a; }}
.box {{ background: #f8fafc; border-left: 5px solid #2563eb; padding: 15px; margin-bottom: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
th {{ background: #1e293b; color: white; }}
</style></head>
<body>
<h1>Raport Biomechaniczny 3D — Analiza Ruchu i Kompensacji</h1>
<div class="box">
  <b>Pacjent:</b> {rekord['pacjent']} &nbsp;|&nbsp;
  <b>Ćwiczenie:</b> {rekord['cwiczenie']} &nbsp;|&nbsp;
  <b>Data:</b> {rekord['data']}<br>
  <b>Źródło:</b> {rekord['zrodlo']} &nbsp;|&nbsp;
  <b>Indeks Jakości Ruchu (MQI):</b> <span style="font-size:1.3em; color:#16a34a;"><b>{rekord['mqi']} / 100%</b></span>
</div>
<h2>1. Ocena wzorców kompensacyjnych (90. percentyl)</h2>
<table><tr><th>Wzorzec kompensacji</th><th>Zmierzony szczyt</th><th>Norma</th><th>Ocena</th></tr>{wiersze_komp}</table>
<h2>2. Zakresy ruchu w stawach 3D (ROM)</h2>
<table><tr><th>Staw / Segment</th><th>Minimum</th><th>Maximum</th><th>Zakres ruchu (ROM)</th></tr>{wiersze_rom}</table>
</body></html>"""
        with open(sciezka_docelowa, "w", encoding="utf-8") as f:
            f.write(html)
    else:
        linie = [
            "=================================================================",
            "       RAPORT BIOMECHANICZNY 3D — ANALIZA KOMPENSACJI RUCHU       ",
            "=================================================================",
            f"Pacjent:                  {rekord['pacjent']}",
            f"Ćwiczenie / Test:         {rekord['cwiczenie']}",
            f"Data badania:             {rekord['data']}",
            f"Źródło nagrania:          {rekord['zrodlo']}",
            f"Przeanalizowane klatki:   {rekord['liczba_klatek']}",
            f"INDEKS JAKOŚCI RUCHU MQI: {rekord['mqi']} / 100%",
            "",
            "--- 1. WYKRYTE KOMPENSACJE RUCHOWE (90. PERCENTYL) ---"
        ]
        if rekord["wykryte_kompensacje"]:
            for k in rekord["wykryte_kompensacje"]:
                linie.append(f" [!] {k['nazwa']}: {k['szczyt']}° (norma <= {k['prog']}°, przekroczenie +{k['przekroczenie']}°)")
        else:
            linie.append(" [OK] Brak istotnych kompensacji — ruch czysty biomechanicznie.")

        linie.append("")
        linie.append("--- 2. ZAKRESY RUCHU STAWÓW 3D (ROM) ---")
        for s in rekord["stawy_rom"]:
            linie.append(f" • {s['staw']:<42} Min: {s['min']:>5}° | Max: {s['max']:>5}° | ROM: {s['rom']:>5}°")

        with open(sciezka_docelowa, "w", encoding="utf-8") as f:
            f.write("\n".join(linie))


def pobierz_historie_pacjenta(id_pacjenta):
    bezpieczny_id = "".join(c for c in id_pacjenta if c.isalnum() or c in ("_", "-")).strip()
    folder = os.path.join(FOLDER_PACJENTOW, bezpieczny_id)
    if not os.path.exists(folder):
        return []

    wyniki = []
    for plik in sorted(os.listdir(folder)):
        if plik.endswith(".json"):
            with open(os.path.join(folder, plik), "r", encoding="utf-8") as f:
                wyniki.append(json.load(f))
    return wyniki