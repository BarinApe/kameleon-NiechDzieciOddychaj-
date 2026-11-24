import os

FOLDER_PROJEKTU = os.path.dirname(os.path.abspath(__file__))
FOLDER_BAZY = os.path.join(FOLDER_PROJEKTU, "baza_cwiczen")
os.makedirs(FOLDER_BAZY, exist_ok=True)

MODELE = {
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

# Parametry nagrywania, uśredniania i automatyzacji
DOMYSLNE_FPS = 20.0
CZAS_NAGRANIA_IDEALU = 10      # sekundy
MIN_KLATEK_PRZED_STOPEM = 35   # ok. 1.5 - 2 sekundy zanim zacznie sprawdzać koniec ruchu
PROG_ZGODNOSCI_KONCA = 14.0    # średnia różnica kątowa (stopnie) uznawana za pozycję końcową
WYMAGANE_KLATKI_KONCA = 8      # ile klatek z rzędu pacjent musi utrzymać pozycję końcową
OKNO_USREDNIANIA_KLATEK = 5    # liczba ostatnich klatek używana do uśredniania drgań

NAZWY_CECH = [
    "Lewy łokieć", "Prawy łokieć",
    "Lewy bark", "Prawy bark",
    "Lewe biodro", "Prawe biodro",
    "Lewe kolano", "Prawe kolano",
    "Lewa kostka", "Prawa kostka",
    "Lewy nadgarstek", "Prawy nadgarstek",
    "Pochylenie szyi", "Skręt głowy",
    "Palce lewej dłoni", "Palce prawej dłoni"
]

PRESETY_WAG = {
    "Równe (wszystkie stawy 1.0)": [1.0] * 16,
    "Nogi i przysiady (kolana, biodra, kostki)": [
        0.3, 0.3, 0.5, 0.5,
        2.0, 2.0, 2.5, 2.5,
        1.8, 1.8, 0.1, 0.1,
        0.6, 0.3, 0.1, 0.1
    ],
    "Ręce i barki (łokcie, barki, nadgarstki)": [
        2.2, 2.2, 2.5, 2.5,
        0.4, 0.4, 0.3, 0.3,
        0.2, 0.2, 1.8, 1.8,
        0.8, 0.4, 1.0, 1.0
    ],
    "Dłonie i palce (nadgarstki i zgięcie palców)": [
        0.8, 0.8, 0.5, 0.5,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 2.2, 2.2,
        0.3, 0.2, 3.0, 3.0
    ],
    "Szyja i głowa (odcinek szyjny)": [
        0.2, 0.2, 1.2, 1.2,
        0.2, 0.2, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        3.0, 3.0, 0.1, 0.1
    ]
}

POLACZENIA_CIALA = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32)
]

POLACZENIA_DLONI = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

# Kontur samej szczęki (żuchwy) od lewego ucha, przez brodę, do prawego ucha
KONTUR_SZCZEKI = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    152,  # środek brody
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454
]