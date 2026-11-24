import os
import math
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from biomechanics_config import FOLDER_PROJEKTU, MODEL_NAZWA, MODEL_URL


class OneEuroFilter3D:
    """
    Adaptacyjny filtr dolnoprzepustowy One-Euro.
    Eliminuje drgania stawów w spoczynku, ale nie wprowadza opóźnienia (lagu) podczas szybkiego ruchu.
    """
    def __init__(self, min_cutoff=1.2, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-5))

    def __call__(self, x, t):
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class AnalizatorRuchu3D:
    def __init__(self):
        self._pobierz_model()
        self.model_path = os.path.join(FOLDER_PROJEKTU, MODEL_NAZWA)
        self.filtr_world = OneEuroFilter3D(min_cutoff=1.0, beta=0.08)
        self.filtr_2d = OneEuroFilter3D(min_cutoff=1.5, beta=0.1)
        self.detector = self._utworz_detektor()
        self.ostatni_ts_ms = -1

    def _utworz_detektor(self):
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6
            )
        )

    def resetuj_sesje(self):
        """Resetuje stan śledzenia wideo przy zmianie źródła (kamera / nowy plik mp4)."""
        self.filtr_world.reset()
        self.filtr_2d.reset()
        self.ostatni_ts_ms = -1
        self.detector.close()
        self.detector = self._utworz_detektor()

    @staticmethod
    def _pobierz_model():
        sciezka = os.path.join(FOLDER_PROJEKTU, MODEL_NAZWA)
        if not os.path.exists(sciezka):
            print(f"Pobieranie modelu biomechanicznego {MODEL_NAZWA}...")
            urllib.request.urlretrieve(MODEL_URL, sciezka)

    @staticmethod
    def kat_3d(a, b, c):
        """Kąt wierzchołkowy w B (0-180°) w przestrzeni 3D."""
        ba = a - b
        bc = c - b
        n_ba = np.linalg.norm(ba)
        n_bc = np.linalg.norm(bc)
        if n_ba < 1e-6 or n_bc < 1e-6:
            return 0.0
        cos_a = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    @staticmethod
    def kat_miedzy_wektorami(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    def przetworz_klatke(self, klatka_rgb, timestamp_ms):
        """
        Główna metoda analizy pojedynczej klatki w trybie VIDEO.
        Zwraca: (punkty_2d_wygl, katy_3d[14], kompensacje_3d[14]) lub (None, None, None).
        """
        if timestamp_ms <= self.ostatni_ts_ms:
            timestamp_ms = self.ostatni_ts_ms + 1
        self.ostatni_ts_ms = timestamp_ms

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=klatka_rgb)
        wynik = self.detector.detect_for_video(mp_img, int(timestamp_ms))

        if not wynik.pose_world_landmarks or not wynik.pose_landmarks:
            return None, None, None

        t_sek = timestamp_ms / 1000.0

        surowe_2d = np.array([[lm.x, lm.y] for lm in wynik.pose_landmarks[0]], dtype=float)
        punkty_2d = self.filtr_2d(surowe_2d, t_sek)

        surowe_3d = np.array([[lm.x, lm.y, lm.z] for lm in wynik.pose_world_landmarks[0]], dtype=float)
        p = self.filtr_world(surowe_3d, t_sek)

        katy = self._oblicz_katy_stawow(p)
        kompensacje = self._oblicz_kompensacje_3d(p)

        return punkty_2d, katy, kompensacje

    def _oblicz_katy_stawow(self, p):
        """Oblicza 14 kluczowych kątów stawowych w przestrzeni 3D (metryczne współrzędne biodrowe)."""
        srodek_barkow = (p[11] + p[12]) * 0.5
        srodek_bioder = (p[23] + p[24]) * 0.5
        srodek_uszu = (p[7] + p[8]) * 0.5

        return np.array([
            self.kat_3d(p[23], p[11], p[13]),  # 0: Bark lewy
            self.kat_3d(p[24], p[12], p[14]),  # 1: Bark prawy
            self.kat_3d(p[11], p[13], p[15]),  # 2: Łokieć lewy
            self.kat_3d(p[12], p[14], p[16]),  # 3: Łokieć prawy
            self.kat_3d(p[13], p[15], p[19]),  # 4: Nadgarstek lewy
            self.kat_3d(p[14], p[16], p[20]),  # 5: Nadgarstek prawy
            self.kat_3d(p[11], p[23], p[25]),  # 6: Biodro lewe
            self.kat_3d(p[12], p[24], p[26]),  # 7: Biodro prawe
            self.kat_3d(p[23], p[25], p[27]),  # 8: Kolano lewe
            self.kat_3d(p[24], p[26], p[28]),  # 9: Kolano prawe
            self.kat_3d(p[25], p[27], p[31]),  # 10: Kostka lewa
            self.kat_3d(p[26], p[28], p[32]),  # 11: Kostka prawa
            self.kat_3d(srodek_bioder, srodek_barkow, srodek_uszu),  # 12: Odcinek szyjny
            self.kat_3d(p[0], srodek_barkow, srodek_bioder),         # 13: Kręgosłup
        ], dtype=float)

    def _oblicz_kompensacje_3d(self, p):
        """
        Wykrywa 14 wzorców kompensacyjnych w przestrzeni 3D (w stopniach):
        W MediaPipe World: X = prawo/lewo, Y = góra/dół (w dół +), Z = przód/tył (do kamery -).
        """
        l_bark, p_bark = p[11], p[12]
        l_biodro, p_biodro = p[23], p[24]
        l_kolano, p_kolano = p[25], p[26]
        l_kostka, p_kostka = p[27], p[28]
        srodek_barkow = (l_bark + p_bark) * 0.5
        srodek_bioder = (l_biodro + p_biodro) * 0.5
        srodek_uszu = (p[7] + p[8]) * 0.5

        wektor_tulowia = srodek_barkow - srodek_bioder
        wys_tulowia = max(abs(wektor_tulowia[1]), 1e-4)

        # 0. Pochylenie boczne tułowia (płaszczyzna czołowa X-Y)
        k_tulow_bok = float(np.degrees(np.arctan2(abs(wektor_tulowia[0]), wys_tulowia)))

        # 1. Pochylenie tułowia przód/tył (płaszczyzna strzałkowa Z-Y)
        k_tulow_przod_tyl = float(np.degrees(np.arctan2(abs(wektor_tulowia[2]), wys_tulowia)))

        # 2. Rotacja osiowa tułowia (kąt między linią barków a linią miednicy w płaszczyźnie poziomej X-Z)
        v_barki_xz = np.array([p_bark[0] - l_bark[0], 0.0, p_bark[2] - l_bark[2]])
        v_miednica_xz = np.array([p_biodro[0] - l_biodro[0], 0.0, p_biodro[2] - l_biodro[2]])
        k_rotacja_tulowia = self.kat_miedzy_wektorami(v_barki_xz, v_miednica_xz)

        # 3. Uniesienie / asymetria barku (Shoulder Hiking)
        v_barki = p_bark - l_bark
        szer_barkow = max(np.hypot(v_barki[0], v_barki[2]), 1e-4)
        k_asymetria_barkow = float(np.degrees(np.arctan2(abs(v_barki[1]), szer_barkow)))

        # 4. Protrakcja barków (wysunięcie barków przed oś tułowia w osi Z)
        przesuniecie_barkow_z = max(0.0, srodek_bioder[2] - srodek_barkow[2])
        k_protrakcja = float(np.degrees(np.arctan2(przesuniecie_barkow_z, wys_tulowia)))

        # 5. Przechylenie boczne miednicy (Pelvic Drop / objaw Trendelenburga)
        v_miednica = p_biodro - l_biodro
        szer_miednicy = max(np.hypot(v_miednica[0], v_miednica[2]), 1e-4)
        k_przechyl_miednicy = float(np.degrees(np.arctan2(abs(v_miednica[1]), szer_miednicy)))

        # 6. Rotacja miednicy w płaszczyźnie poprzecznej X-Z względem osi stóp
        v_stopy_xz = np.array([p_kostka[0] - l_kostka[0], 0.0, p_kostka[2] - l_kostka[2]])
        k_rotacja_miednicy = self.kat_miedzy_wektorami(v_miednica_xz, v_stopy_xz)

        # 7-10. Koślawienie (Valgus) i Szpotawienie (Varus) kolan w płaszczyźnie czołowej
        def analiza_osi_kolana(biodro, kolano, kostka, czy_lewa):
            dy = kostka[1] - biodro[1]
            if abs(dy) < 1e-4:
                return 0.0, 0.0
            t = np.clip((kolano[1] - biodro[1]) / dy, 0.0, 1.0)
            os_x = biodro[0] + t * (kostka[0] - biodro[0])
            dl_uda = max(abs(kolano[1] - biodro[1]), 1e-4)
            roznica_do_srodka = (os_x - kolano[0]) if czy_lewa else (kolano[0] - os_x)
            kat = float(np.degrees(np.arctan2(abs(roznica_do_srodka), dl_uda)))
            if roznica_do_srodka > 0:
                return kat, 0.0
            else:
                return 0.0, kat

        valgus_l, varus_l = analiza_osi_kolana(l_biodro, l_kolano, l_kostka, True)
        valgus_p, varus_p = analiza_osi_kolana(p_biodro, p_kolano, p_kostka, False)

        # 11-12. Przeprost kolan (Hyperextension w osi Z-Y)
        def przeprost_kolana(biodro, kolano, kostka):
            dy = kostka[1] - biodro[1]
            if abs(dy) < 1e-4:
                return 0.0
            t = np.clip((kolano[1] - biodro[1]) / dy, 0.0, 1.0)
            os_z = biodro[2] + t * (kostka[2] - biodro[2])
            przeprost_z = max(0.0, kolano[2] - os_z)
            dl_uda = max(abs(kolano[1] - biodro[1]), 1e-4)
            return float(np.degrees(np.arctan2(przeprost_z, dl_uda)))

        przeprost_l = przeprost_kolana(l_biodro, l_kolano, l_kostka)
        przeprost_p = przeprost_kolana(p_biodro, p_kolano, p_kostka)

        # 13. Wysunięcie głowy w przód (Forward Head Posture - FHP)
        wys_szyi = max(abs(srodek_barkow[1] - srodek_uszu[1]), 1e-4)
        wysuniecie_glowy_z = max(0.0, srodek_barkow[2] - srodek_uszu[2])
        k_fhp = float(np.degrees(np.arctan2(wysuniecie_glowy_z, wys_szyi)))

        return np.array([
            k_tulow_bok,
            k_tulow_przod_tyl,
            k_rotacja_tulowia,
            k_asymetria_barkow,
            k_protrakcja,
            k_przechyl_miednicy,
            k_rotacja_miednicy,
            valgus_l,
            valgus_p,
            varus_l,
            varus_p,
            przeprost_l,
            przeprost_p,
            k_fhp
        ], dtype=float)
