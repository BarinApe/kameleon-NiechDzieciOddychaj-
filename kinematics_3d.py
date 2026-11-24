import os
import math
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from biomechanics_config import FOLDER_PROJEKTU, MODEL_NAZWA, MODEL_URL


class OneEuroFilter3D:
    """Adaptacyjny filtr dolnoprzepustowy One-Euro eliminujący drgania bez opóźnienia (lagu)."""
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
        self.fallback_detector = self._utworz_detektor_image()
        self.ostatni_ts_ms = -1

    def _utworz_detektor(self):
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
        )

    def _utworz_detektor_image(self):
        """Zapasowy detektor pełnoklatkowy na wypadek zgubienia śledzenia w trybie VIDEO."""
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3
            )
        )

    def resetuj_sesje(self):
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
        if timestamp_ms <= self.ostatni_ts_ms:
            timestamp_ms = self.ostatni_ts_ms + 1
        self.ostatni_ts_ms = timestamp_ms

        # Wymuszenie ciągłości tablicy uint8 w pamięci (kluczowe dla MediaPipe C++)
        klatka_czysta = np.ascontiguousarray(klatka_rgb, dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=klatka_czysta)

        wynik = self.detector.detect_for_video(mp_img, int(timestamp_ms))

        # Jeśli tracker VIDEO nie złapał jeszcze sylwetki, użyj detekcji pełnoklatkowej
        if not wynik.pose_world_landmarks or not wynik.pose_landmarks:
            wynik = self.fallback_detector.detect(mp_img)

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
        srodek_barkow = (p[11] + p[12]) * 0.5
        srodek_bioder = (p[23] + p[24]) * 0.5
        srodek_uszu = (p[7] + p[8]) * 0.5

        return np.array([
            self.kat_3d(p[23], p[11], p[13]),
            self.kat_3d(p[24], p[12], p[14]),
            self.kat_3d(p[11], p[13], p[15]),
            self.kat_3d(p[12], p[14], p[16]),
            self.kat_3d(p[13], p[15], p[19]),
            self.kat_3d(p[14], p[16], p[20]),
            self.kat_3d(p[11], p[23], p[25]),
            self.kat_3d(p[12], p[24], p[26]),
            self.kat_3d(p[23], p[25], p[27]),
            self.kat_3d(p[24], p[26], p[28]),
            self.kat_3d(p[25], p[27], p[31]),
            self.kat_3d(p[26], p[28], p[32]),
            self.kat_3d(srodek_bioder, srodek_barkow, srodek_uszu),
            self.kat_3d(p[0], srodek_barkow, srodek_bioder),
        ], dtype=float)

    def _oblicz_kompensacje_3d(self, p):
        l_bark, p_bark = p[11], p[12]
        l_biodro, p_biodro = p[23], p[24]
        l_kolano, p_kolano = p[25], p[26]
        l_kostka, p_kostka = p[27], p[28]
        srodek_barkow = (l_bark + p_bark) * 0.5
        srodek_bioder = (l_biodro + p_biodro) * 0.5
        srodek_uszu = (p[7] + p[8]) * 0.5

        wektor_tulowia = srodek_barkow - srodek_bioder
        wys_tulowia = max(abs(wektor_tulowia[1]), 1e-4)

        k_tulow_bok = float(np.degrees(np.arctan2(abs(wektor_tulowia[0]), wys_tulowia)))
        k_tulow_przod_tyl = float(np.degrees(np.arctan2(abs(wektor_tulowia[2]), wys_tulowia)))

        v_barki_xz = np.array([p_bark[0] - l_bark[0], 0.0, p_bark[2] - l_bark[2]])
        v_miednica_xz = np.array([p_biodro[0] - l_biodro[0], 0.0, p_biodro[2] - l_biodro[2]])
        k_rotacja_tulowia = self.kat_miedzy_wektorami(v_barki_xz, v_miednica_xz)

        v_barki = p_bark - l_bark
        szer_barkow = max(np.hypot(v_barki[0], v_barki[2]), 1e-4)
        k_asymetria_barkow = float(np.degrees(np.arctan2(abs(v_barki[1]), szer_barkow)))

        przesuniecie_barkow_z = max(0.0, srodek_bioder[2] - srodek_barkow[2])
        k_protrakcja = float(np.degrees(np.arctan2(przesuniecie_barkow_z, wys_tulowia)))

        v_miednica = p_biodro - l_biodro
        szer_miednicy = max(np.hypot(v_miednica[0], v_miednica[2]), 1e-4)
        k_przechyl_miednicy = float(np.degrees(np.arctan2(abs(v_miednica[1]), szer_miednicy)))

        v_stopy_xz = np.array([p_kostka[0] - l_kostka[0], 0.0, p_kostka[2] - l_kostka[2]])
        k_rotacja_miednicy = self.kat_miedzy_wektorami(v_miednica_xz, v_stopy_xz)

        def analiza_osi_kolana(biodro, kolano, kostka, czy_lewa):
            dy = kostka[1] - biodro[1]
            if abs(dy) < 1e-4:
                return 0.0, 0.0
            t = np.clip((kolano[1] - biodro[1]) / dy, 0.0, 1.0)
            os_x = biodro[0] + t * (kostka[0] - biodro[0])
            dl_uda = max(abs(kolano[1] - biodro[1]), 1e-4)
            roznica_do_srodka = (os_x - kolano[0]) if czy_lewa else (kolano[0] - os_x)
            kat = float(np.degrees(np.arctan2(abs(roznica_do_srodka), dl_uda)))
            return (kat, 0.0) if roznica_do_srodka > 0 else (0.0, kat)

        valgus_l, varus_l = analiza_osi_kolana(l_biodro, l_kolano, l_kostka, True)
        valgus_p, varus_p = analiza_osi_kolana(p_biodro, p_kolano, p_kostka, False)

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

        wys_szyi = max(abs(srodek_barkow[1] - srodek_uszu[1]), 1e-4)
        wysuniecie_glowy_z = max(0.0, srodek_barkow[2] - srodek_uszu[2])
        k_fhp = float(np.degrees(np.arctan2(wysuniecie_glowy_z, wys_szyi)))

        return np.array([
            k_tulow_bok, k_tulow_przod_tyl, k_rotacja_tulowia,
            k_asymetria_barkow, k_protrakcja, k_przechyl_miednicy,
            k_rotacja_miednicy, valgus_l, valgus_p,
            varus_l, varus_p, przeprost_l, przeprost_p, k_fhp
        ], dtype=float)