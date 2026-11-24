import os
import json
from datetime import datetime
import numpy as np

from biomechanics_config import FOLDER_PACJENTOW, NAZWY_KATOW_3D, KOMPENSACJE_DEFINICJE


def analizuj_pelna_sekwencje(historia_katow, historia_kompensacji):
    """
    Przetwarza zebrane dane 3D z całej próby pacjenta (pliku MP4 lub nagrania na żywo).
    Wyznacza:
    - Zakresy ruchu (ROM: Min, Max, Amplituda) dla każdego z 14 stawów
    - Szczytowe (90. percentyl) oraz średnie wartości dla 14 kompensacji
    - Globalny Indeks Jakości Ruchu (MQI: 0-100%)
    """
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


def zapisz_wynik_pacjenta(id_pacjenta, nazwa_cwiczenia, zrodlo, wyniki_analizy):
    """Zapisuje badanie do historii pacjenta w celu śledzenia postępów rehabilitacji."""
    bezpieczny_id = "".join(c for c in id_pacjenta if c.isalnum() or c in ("_", "-")).strip() or "Pacjent_01"
    folder = os.path.join(FOLDER_PACJENTOW, bezpieczny_id)
    os.makedirs(folder, exist_ok=True)

    znacznik = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sciezka = os.path.join(folder, f"{znacznik}_{nazwa_cwiczenia}.json")

    rekord = {
        "pacjent": bezpieczny_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cwiczenie": nazwa_cwiczenia,
        "zrodlo": zrodlo,
        "mqi": wyniki_analizy["mqi"],
        "wykryte_kompensacje": wyniki_analizy["wykryte_kompensacje"],
        "stawy_rom": wyniki_analizy["stawy_rom"]
    }

    with open(sciezka, "w", encoding="utf-8") as f:
        json.dump(rekord, f, indent=2, ensure_ascii=False)
    return sciezka


def pobierz_historie_pacjenta(id_pacjenta):
    """Zwraca chronologiczną listę badań pacjenta do porównania postępów."""
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
