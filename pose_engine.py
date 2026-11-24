import os
import urllib.request
from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from config import (
    FOLDER_PROJEKTU, MODELE, OKNO_USREDNIANIA_KLATEK,
    POLACZENIA_CIALA, POLACZENIA_DLONI, KONTUR_SZCZEKI
)


class SilnikSzkieletu:
    def __init__(self):
        self._pobierz_brakujace_modele()

        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=os.path.join(FOLDER_PROJEKTU, "pose_landmarker_full.task")
                ),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.65,
                min_tracking_confidence=0.65
            )
        )

        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=os.path.join(FOLDER_PROJEKTU, "hand_landmarker.task")
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.55
            )
        )

        # Podwyższona pewność detekcji twarzy dla stabilnego konturu szczęki
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=os.path.join(FOLDER_PROJEKTU, "face_landmarker.task")
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.7,
                min_face_presence_confidence=0.7,
                min_tracking_confidence=0.7
            )
        )

        # Bufory 5-klatkowe (średnia krocząca eliminująca drgania)
        self.bufor_pose_2d = deque(maxlen=OKNO_USREDNIANIA_KLATEK)
        self.bufor_szczeka_2d = deque(maxlen=OKNO_USREDNIANIA_KLATEK)
        self.bufor_cechy_3d = deque(maxlen=OKNO_USREDNIANIA_KLATEK)

    def resetuj_filtr(self):
        """Czyści bufory 5-klatkowe przy rozpoczęciu nowego nagrania lub pliku wideo."""
        self.bufor_pose_2d.clear()
        self.bufor_szczeka_2d.clear()
        self.bufor_cechy_3d.clear()

    @staticmethod
    def _pobierz_brakujace_modele():
        for nazwa_pliku, url in MODELE.items():
            sciezka = os.path.join(FOLDER_PROJEKTU, nazwa_pliku)
            if not os.path.exists(sciezka):
                print(f"Pobieranie modelu {nazwa_pliku}...")
                urllib.request.urlretrieve(url, sciezka)

    @staticmethod
    def oblicz_kat_3d(a, b, c):
        """Oblicza dokładny kąt (0-180°) w przestrzeni 3D w wierzchołku b."""
        a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
        ba = a - b
        bc = c - b
        norma_ba = np.linalg.norm(ba)
        norma_bc = np.linalg.norm(bc)
        if norma_ba < 1e-7 or norma_bc < 1e-7:
            return 0.0
        cosinus = np.clip(np.dot(ba, bc) / (norma_ba * norma_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosinus)))

    @staticmethod
    def srodek_odcinka(p1, p2):
        return [(p1[i] + p2[i]) / 2.0 for i in range(len(p1))]

    def _oblicz_zgiecie_dloni_3d(self, hand_world_landmarks):
        def hp(idx):
            lm = hand_world_landmarks[idx]
            return [lm.x, lm.y, lm.z]

        katy_palcow = [
            self.oblicz_kat_3d(hp(5), hp(6), hp(8)),
            self.oblicz_kat_3d(hp(9), hp(10), hp(12)),
            self.oblicz_kat_3d(hp(13), hp(14), hp(16)),
            self.oblicz_kat_3d(hp(17), hp(18), hp(20))
        ]
        return float(np.mean(katy_palcow))

    def analizuj_klatke(self, obraz_rgb):
        """
        Analizuje klatkę, uśrednia wyniki z ostatnich 5 klatek,
        rysuje szkielet + kontur szczęki i zwraca uśredniony wektor 16 kątów 3D.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=obraz_rgb)

        pose_result = self.pose_detector.detect(mp_image)
        hands_result = self.hand_detector.detect(mp_image)
        face_result = self.face_detector.detect(mp_image)

        # 1. Uśrednianie 5-klatkowe dla szkieletu 2D ciała
        wygladzony_szkielet_2d = None
        if pose_result.pose_landmarks:
            surowe_2d = np.array([[lm.x, lm.y] for lm in pose_result.pose_landmarks[0]])
            self.bufor_pose_2d.append(surowe_2d)
            wygladzony_szkielet_2d = np.mean(self.bufor_pose_2d, axis=0)
        else:
            self.bufor_pose_2d.clear()

        # 2. Uśrednianie 5-klatkowe dla punktów konturu szczęki (poniżej uszu)
        wygladzona_szczeka_2d = None
        if face_result.face_landmarks:
            face_lm = face_result.face_landmarks[0]
            surowa_szczeka = np.array([[face_lm[i].x, face_lm[i].y] for i in KONTUR_SZCZEKI])
            self.bufor_szczeka_2d.append(surowa_szczeka)
            wygladzona_szczeka_2d = np.mean(self.bufor_szczeka_2d, axis=0)
        else:
            self.bufor_szczeka_2d.clear()

        self._rysuj_sylwetke(obraz_rgb, wygladzony_szkielet_2d, hands_result, wygladzona_szczeka_2d)

        # 3. Uśrednianie 5-klatkowe dla wektora 16 kątów 3D
        if not pose_result.pose_world_landmarks:
            self.bufor_cechy_3d.clear()
            return None

        surowe_cechy = np.array(
            self._wyciagnij_cechy_3d(pose_result.pose_world_landmarks[0], hands_result),
            dtype=float
        )
        self.bufor_cechy_3d.append(surowe_cechy)
        usrednione_cechy = np.mean(self.bufor_cechy_3d, axis=0)

        return usrednione_cechy.tolist()

    def _wyciagnij_cechy_3d(self, pose_world_lm, hands_result):
        def p(idx):
            lm = pose_world_lm[idx]
            return [lm.x, lm.y, lm.z]

        lewy_bark, prawy_bark = p(11), p(12)
        lewe_ucho, prawe_ucho = p(7), p(8)
        nos = p(0)

        podstawa_szyi = self.srodek_odcinka(lewy_bark, prawy_bark)
        srodek_uszu = self.srodek_odcinka(lewe_ucho, prawe_ucho)

        zgiecie_lewa_dlon = 180.0
        zgiecie_prawa_dlon = 180.0

        zrodlo_dloni = hands_result.hand_world_landmarks if (hands_result and hands_result.hand_world_landmarks) else (
            hands_result.hand_landmarks if hands_result else None
        )

        if zrodlo_dloni:
            for idx, hand_lm in enumerate(zrodlo_dloni):
                zgiecie = self._oblicz_zgiecie_dloni_3d(hand_lm)
                if hands_result.handedness and idx < len(hands_result.handedness):
                    strona = hands_result.handedness[idx][0].category_name
                    if strona == "Left":
                        zgiecie_prawa_dlon = zgiecie
                    else:
                        zgiecie_lewa_dlon = zgiecie

        return [
            self.oblicz_kat_3d(p(11), p(13), p(15)),  # Lewy łokieć
            self.oblicz_kat_3d(p(12), p(14), p(16)),  # Prawy łokieć
            self.oblicz_kat_3d(p(23), p(11), p(13)),  # Lewy bark
            self.oblicz_kat_3d(p(24), p(12), p(14)),  # Prawy bark
            self.oblicz_kat_3d(p(11), p(23), p(25)),  # Lewe biodro
            self.oblicz_kat_3d(p(12), p(24), p(26)),  # Prawe biodro
            self.oblicz_kat_3d(p(23), p(25), p(27)),  # Lewe kolano
            self.oblicz_kat_3d(p(24), p(26), p(28)),  # Prawe kolano
            self.oblicz_kat_3d(p(25), p(27), p(31)),  # Lewa kostka
            self.oblicz_kat_3d(p(26), p(28), p(32)),  # Prawa kostka
            self.oblicz_kat_3d(p(13), p(15), p(19)),  # Lewy nadgarstek
            self.oblicz_kat_3d(p(14), p(16), p(20)),  # Prawy nadgarstek
            self.oblicz_kat_3d(prawy_bark, podstawa_szyi, srodek_uszu),  # Pochylenie szyi
            self.oblicz_kat_3d(lewe_ucho, nos, prawe_ucho),              # Skręt głowy
            zgiecie_lewa_dlon,
            zgiecie_prawa_dlon
        ]

    @staticmethod
    def _rysuj_sylwetke(obraz_rgb, pose_2d, hands_result, szczeka_2d):
        h, w, _ = obraz_rgb.shape

        # 1. Kontur samej szczęki poniżej uszu (linia + wyraźne punkty węzłowe)
        if szczeka_2d is not None:
            pts = np.array(
                [[int(pt[0] * w), int(pt[1] * h)] for pt in szczeka_2d],
                dtype=np.int32
            )
            cv2.polylines(obraz_rgb, [pts], isClosed=False, color=(100, 220, 255), thickness=2, lineType=cv2.LINE_AA)
            for idx_pt, (px, py) in enumerate(pts):
                # Środek brody wyróżniony nieco większym punktem
                promien = 4 if idx_pt == len(pts) // 2 else 3
                kolor = (255, 255, 0) if idx_pt == len(pts) // 2 else (0, 230, 255)
                cv2.circle(obraz_rgb, (px, py), promien, kolor, -1, cv2.LINE_AA)

        # 2. Szkielet ciała + szyja + uszy (uśredniony z 5 klatek)
        if pose_2d is not None:
            punkty = {idx: (int(pt[0] * w), int(pt[1] * h)) for idx, pt in enumerate(pose_2d)}

            for a, b in POLACZENIA_CIALA:
                if a in punkty and b in punkty:
                    cv2.line(obraz_rgb, punkty[a], punkty[b], (255, 255, 255), 2, cv2.LINE_AA)

            if all(k in punkty for k in (7, 8, 11, 12)):
                podstawa_szyi = (
                    (punkty[11][0] + punkty[12][0]) // 2,
                    (punkty[11][1] + punkty[12][1]) // 2
                )
                srodek_uszu = (
                    (punkty[7][0] + punkty[8][0]) // 2,
                    (punkty[7][1] + punkty[8][1]) // 2
                )
                cv2.line(obraz_rgb, punkty[7], punkty[8], (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(obraz_rgb, podstawa_szyi, srodek_uszu, (255, 215, 0), 3, cv2.LINE_AA)
                cv2.circle(obraz_rgb, podstawa_szyi, 5, (255, 215, 0), -1)
                cv2.circle(obraz_rgb, punkty[7], 5, (0, 255, 255), -1)
                cv2.circle(obraz_rgb, punkty[8], 5, (0, 255, 255), -1)

            for idx in range(11, 33):
                if idx in punkty:
                    cv2.circle(obraz_rgb, punkty[idx], 4, (0, 255, 0), -1)

        # 3. Palce i dłonie
        if hands_result and hands_result.hand_landmarks:
            for hand_lm in hands_result.hand_landmarks:
                hp = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(hand_lm)}
                for a, b in POLACZENIA_DLONI:
                    cv2.line(obraz_rgb, hp[a], hp[b], (255, 165, 0), 2, cv2.LINE_AA)
                for i, pt in hp.items():
                    kolor = (255, 50, 50) if i in (4, 8, 12, 16, 20) else (255, 255, 100)
                    cv2.circle(obraz_rgb, pt, 3, kolor, -1)


def rysuj_tekst_z_obrysem(obraz, tekst, pozycja, skala=1.0, kolor=(255, 255, 255), grubosc=2):
    cv2.putText(obraz, tekst, pozycja, cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), grubosc + 3, cv2.LINE_AA)
    cv2.putText(obraz, tekst, pozycja, cv2.FONT_HERSHEY_SIMPLEX, skala, kolor, grubosc, cv2.LINE_AA)