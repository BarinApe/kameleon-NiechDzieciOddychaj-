import os
import json
import shutil
from datetime import datetime
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

from config import FOLDER_BAZY, DOMYSLNE_FPS, NAZWY_CECH


def pobierz_liste_cwiczen():
    wynik = []
    for d in sorted(os.listdir(FOLDER_BAZY)):
        sciezka_dir = os.path.join(FOLDER_BAZY, d)
        if os.path.isdir(sciezka_dir):
            ma_ideal = os.path.exists(os.path.join(sciezka_dir, "ideal_dane.json"))
            wynik.append((d, ma_ideal))
    return wynik


def utworz_cwiczenie(nazwa):
    bezpieczna = "".join(c for c in nazwa if c.isalnum() or c in (" ", "_", "-")).strip()
    if not bezpieczna:
        return None
    folder = os.path.join(FOLDER_BAZY, bezpieczna)
    os.makedirs(os.path.join(folder, "probki"), exist_ok=True)
    # Domyślne równe wagi 1.0 dla wszystkich 16 cech
    zapisz_wagi_cwiczenia(bezpieczna, [1.0] * len(NAZWY_CECH))
    return bezpieczna


def usun_cwiczenie(nazwa):
    shutil.rmtree(os.path.join(FOLDER_BAZY, nazwa), ignore_errors=True)


# --- ZARZĄDZANIE WAGAMI STAWÓW ---

def zapisz_wagi_cwiczenia(nazwa_cwiczenia, lista_wag):
    sciezka = os.path.join(FOLDER_BAZY, nazwa_cwiczenia, "wagi.json")
    os.makedirs(os.path.dirname(sciezka), exist_ok=True)
    with open(sciezka, "w", encoding="utf-8") as f:
        json.dump(lista_wag, f, indent=2)


def wczytaj_wagi_cwiczenia(nazwa_cwiczenia):
    sciezka = os.path.join(FOLDER_BAZY, nazwa_cwiczenia, "wagi.json")
    if os.path.exists(sciezka):
        with open(sciezka, "r", encoding="utf-8") as f:
            wagi = json.load(f)
            if len(wagi) == len(NAZWY_CECH):
                return wagi
    return [1.0] * len(NAZWY_CECH)


# --- ZAPIS WIDEO I WZORCÓW ---

def zapisz_wideo(sciezka_pliku, klatki_rgb, fps=DOMYSLNE_FPS):
    if not klatki_rgb:
        return
    h, w, _ = klatki_rgb[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(sciezka_pliku, fourcc, fps, (w, h))
    for klatka in klatki_rgb:
        out.write(cv2.cvtColor(klatka, cv2.COLOR_RGB2BGR))
    out.release()


def zapisz_ideal(nazwa_cwiczenia, sekwencja, klatki_rgb, fps=DOMYSLNE_FPS):
    folder_cw = os.path.join(FOLDER_BAZY, nazwa_cwiczenia)
    os.makedirs(folder_cw, exist_ok=True)
    with open(os.path.join(folder_cw, "ideal_dane.json"), "w", encoding="utf-8") as f:
        json.dump(sekwencja, f)
    zapisz_wideo(os.path.join(folder_cw, "ideal_wideo.mp4"), klatki_rgb, fps)


def wczytaj_ideal(nazwa_cwiczenia):
    sciezka = os.path.join(FOLDER_BAZY, nazwa_cwiczenia, "ideal_dane.json")
    if not os.path.exists(sciezka):
        return []
    with open(sciezka, "r", encoding="utf-8") as f:
        return json.load(f)


# --- WAŻONA OCENA DTW ---

def ocen_ruch_szczegolowo(ideal_dane, probka_dane, wagi=None):
    """
    Porównuje sekwencje za pomocą ważonego DTW (priorytetyzując ważne stawy dla danego ćwiczenia).
    Zwraca: (wynik_procent, lista_najwiekszych_bledow).
    """
    if (
        not ideal_dane or not probka_dane
        or len(ideal_dane[0]) != len(probka_dane[0])
    ):
        return 0.0, []

    liczba_cech = len(ideal_dane[0])
    if wagi is None or len(wagi) != liczba_cech:
        wagi = [1.0] * liczba_cech

    wagi_arr = np.array(wagi, dtype=float)
    # Skalujemy wektory przez pierwiastek z wag, aby odległość euklidesowa uwzględniała wagi
    pierwiastki_wag = np.sqrt(np.maximum(wagi_arr, 0.0))

    ideal_wazony = [np.array(k) * pierwiastki_wag for k in ideal_dane]
    probka_wazona = [np.array(k) * pierwiastki_wag for k in probka_dane]

    dystans, sciezka_dopasowania = fastdtw(ideal_wazony, probka_wazona, dist=euclidean)
    dlugosc = max(len(ideal_dane), len(probka_dane))

    # Normalizacja względem sumy wag, aby skala procentowa była spójna niezależnie od wag
    wspolczynnik_skali = np.sqrt(np.sum(wagi_arr) / liczba_cech) if np.sum(wagi_arr) > 0 else 1.0
    sredni_blad = (dystans / dlugosc) / max(wspolczynnik_skali, 0.1)
    ocena = max(0.0, min(100.0, 100.0 - (sredni_blad / 2.2)))

    # Obliczenie średniego błędu dla każdego stawu (uwzględniając wagi przy rankingu błędów)
    surowe_roznice = np.zeros(liczba_cech)
    for idx_ideal, idx_probka in sciezka_dopasowania:
        surowe_roznice += np.abs(np.array(ideal_dane[idx_ideal]) - np.array(probka_dane[idx_probka]))
    surowe_roznice /= len(sciezka_dopasowania)

    wazony_ranking = surowe_roznice * wagi_arr
    indeksy_posortowane = np.argsort(wazony_ranking)[::-1]

    najwieksze_bledy = [
        (NAZWY_CECH[i], round(float(surowe_roznice[i]), 1))
        for i in indeksy_posortowane[:3]
        if i < len(NAZWY_CECH) and surowe_roznice[i] > 8.0 and wagi_arr[i] > 0.2
    ]

    return round(ocena, 1), najwieksze_bledy


def zapisz_probke_pacjenta(nazwa_cwiczenia, znacznik_czasu, sekwencja, klatki_rgb, ocena, bledy, fps=DOMYSLNE_FPS, prefiks="probka"):
    folder_probek = os.path.join(FOLDER_BAZY, nazwa_cwiczenia, "probki")
    os.makedirs(folder_probek, exist_ok=True)

    sciezka_wideo = os.path.join(folder_probek, f"{prefiks}_{znacznik_czasu}.mp4")
    sciezka_json = os.path.join(folder_probek, f"{prefiks}_{znacznik_czasu}.json")

    zapisz_wideo(sciezka_wideo, klatki_rgb, fps)
    with open(sciezka_json, "w", encoding="utf-8") as f:
        json.dump({
            "cwiczenie": nazwa_cwiczenia,
            "data": znacznik_czasu,
            "wynik_procent": ocena,
            "najwieksze_odchylenia_stopnie": bledy,
            "sekwencja_katow": sekwencja
        }, f, indent=2, ensure_ascii=False)


# --- GENERATOR RAPORTU PDF Z WYKRESEM POSTĘPÓW ---

def _usun_polskie_znaki(tekst):
    """Bezpieczna konwersja znaków dla standardowych czcionek PDF."""
    mapa = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    return tekst.translate(mapa)


def generuj_raport_pdf(nazwa_cwiczenia):
    """
    Zbiera wszystkie zapisane pliki JSON z folderu 'probki' danego ćwiczenia,
    generuje wykres postępów i zapisuje raport PDF. Zwraca ścieżkę do PDF lub None.
    """
    folder_cw = os.path.join(FOLDER_BAZY, nazwa_cwiczenia)
    folder_probek = os.path.join(folder_cw, "probki")
    if not os.path.exists(folder_probek):
        return None

    pliki_json = sorted([f for f in os.listdir(folder_probek) if f.endswith(".json")])
    if not pliki_json:
        return None

    historia = []
    for plik in pliki_json:
        with open(os.path.join(folder_probek, plik), "r", encoding="utf-8") as f:
            dane = json.load(f)
            historia.append({
                "data": dane.get("data", plik),
                "wynik": float(dane.get("wynik_procent", 0.0)),
                "bledy": dane.get("najwieksze_odchylenia_stopnie", [])
            })

    # 1. Generowanie wykresu postępów (Matplotlib)
    daty = [h["data"].replace("_", "\n") for h in historia]
    wyniki = [h["wynik"] for h in historia]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(range(len(wyniki)), wyniki, marker="o", linewidth=2.5, color="#1565c0", markersize=7)
    ax.fill_between(range(len(wyniki)), wyniki, color="#bbdefb", alpha=0.4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Zgodnosc z idealem (%)")
    ax.set_title(f"Postepy pacjenta - {_usun_polskie_znaki(nazwa_cwiczenia)}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(daty)))
    ax.set_xticklabels(daty, fontsize=8)

    for i, v in enumerate(wyniki):
        ax.annotate(f"{v:.1f}%", (i, v + 3), ha="center", fontsize=8, fontweight="bold")

    sciezka_wykresu = os.path.join(folder_cw, "temp_wykres.png")
    plt.tight_layout()
    plt.savefig(sciezka_wykresu, dpi=150)
    plt.close(fig)

    # 2. Tworzenie dokumentu PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _usun_polskie_znaki(f"Raport Rehabilitacji: {nazwa_cwiczenia}"), ln=True)

    pdf.set_font("Helvetica", "", 10)
    teraz = datetime.now().strftime("%Y-%m-%d %H:%M")
    srednia = sum(wyniki) / len(wyniki)
    najlepszy = max(wyniki)
    pdf.cell(0, 7, f"Data wygenerowania raportu: {teraz}", ln=True)
    pdf.cell(0, 7, f"Liczba wykonanych prob: {len(historia)} | Sredni wynik: {srednia:.1f}% | Najlepszy wynik: {najlepszy:.1f}%", ln=True)
    pdf.ln(4)

    # Wstawienie wykresu
    pdf.image(sciezka_wykresu, x=10, w=190)
    pdf.ln(6)

    # Tabela historii sesji
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Szczegolowa historia prob:", ln=True)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 238, 248)
    pdf.cell(45, 7, "Data / Czas", border=1, fill=True)
    pdf.cell(25, 7, "Wynik", border=1, fill=True, align="C")
    pdf.cell(120, 7, "Wykryte odchylenia stawow", border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 9)
    for wpis in historia:
        tekst_bledow = ", ".join([f"{s} ({k} st.)" for s, k in wpis["bledy"]]) if wpis["bledy"] else "Brak istotnych odchylen"
        pdf.cell(45, 7, wpis["data"], border=1)
        pdf.cell(25, 7, f"{wpis['wynik']:.1f}%", border=1, align="C")
        pdf.cell(120, 7, _usun_polskie_znaki(tekst_bledow)[:68], border=1, ln=True)

    znacznik = datetime.now().strftime("%Y-%m-%d_%H-%M")
    sciezka_pdf = os.path.join(folder_cw, f"Raport_{nazwa_cwiczenia}_{znacznik}.pdf")
    pdf.output(sciezka_pdf)

    if os.path.exists(sciezka_wykresu):
        os.remove(sciezka_wykresu)

    return sciezka_pdf