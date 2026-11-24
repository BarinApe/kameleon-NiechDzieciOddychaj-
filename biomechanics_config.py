import os

FOLDER_PROJEKTU = os.path.dirname(os.path.abspath(__file__))
FOLDER_PACJENTOW = os.path.join(FOLDER_PROJEKTU, "pacjenci_dane")
os.makedirs(FOLDER_PACJENTOW, exist_ok=True)

# Używamy jednego, najdokładniejszego modelu Pose Heavy w trybie VIDEO (szybszy i dokładniejszy niż 3 modele w IMAGE)
MODEL_NAZWA = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# 14 precyzyjnych kątów stawowych w przestrzeni 3D (w stopniach)
NAZWY_KATOW_3D = [
    "Bark lewy (odwiedzenie/zgięcie)",
    "Bark prawy (odwiedzenie/zgięcie)",
    "Łokieć lewy (zgięcie/wyprost)",
    "Łokieć prawy (zgięcie/wyprost)",
    "Nadgarstek lewy (odchylenie)",
    "Nadgarstek prawy (odchylenie)",
    "Biodro lewe (zgięcie/wyprost)",
    "Biodro prawe (zgięcie/wyprost)",
    "Kolano lewe (zgięcie/wyprost)",
    "Kolano prawe (zgięcie/wyprost)",
    "Staw skokowy lewy (zgięcie grzbietowe)",
    "Staw skokowy prawy (zgięcie grzbietowe)",
    "Odcinek szyjny (pochylenie)",
    "Kręgosłup lędźwiowo-piersiowy (zgięcie)"
]

# 14 wektorowych kompensacji ruchowych w 3D wraz z progami fizjologicznymi (stopnie / cm)
KOMPENSACJE_DEFINICJE = [
    {"nazwa": "Pochylenie boczne tułowia (Lateral Lean)", "prog": 6.5, "jednostka": "°"},
    {"nazwa": "Pochylenie tułowia przód/tył (Trunk Flex/Ext)", "prog": 10.0, "jednostka": "°"},
    {"nazwa": "Rotacja osiowa tułowia (Trunk Rotation)", "prog": 8.0, "jednostka": "°"},
    {"nazwa": "Asymetria / uniesienie barku (Shoulder Hiking)", "prog": 5.5, "jednostka": "°"},
    {"nazwa": "Protrakcja barków (zaokrąglenie pleców)", "prog": 12.0, "jednostka": "°"},
    {"nazwa": "Przechylenie boczne miednicy (Pelvic Drop/Trendelenburg)", "prog": 5.0, "jednostka": "°"},
    {"nazwa": "Rotacja miednicy w płaszczyźnie poprzecznej", "prog": 8.0, "jednostka": "°"},
    {"nazwa": "Koślawienie lewego kolana (Knee Valgus L)", "prog": 7.5, "jednostka": "°"},
    {"nazwa": "Koślawienie prawego kolana (Knee Valgus R)", "prog": 7.5, "jednostka": "°"},
    {"nazwa": "Szpotawienie lewego kolana (Knee Varus L)", "prog": 8.0, "jednostka": "°"},
    {"nazwa": "Szpotawienie prawego kolana (Knee Varus R)", "prog": 8.0, "jednostka": "°"},
    {"nazwa": "Przeprost lewego kolana (Hyperextension L)", "prog": 4.0, "jednostka": "°"},
    {"nazwa": "Przeprost prawego kolana (Hyperextension R)", "prog": 4.0, "jednostka": "°"},
    {"nazwa": "Wysunięcie głowy w przód (Forward Head Posture)", "prog": 14.0, "jednostka": "°"},
]

POLACZENIA_SZKIELETU = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
    (15, 17), (15, 19), (16, 18), (16, 20)
]
