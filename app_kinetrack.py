import os
import time
import math
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from biomechanics_config import FOLDER_PACJENTOW, NAZWY_KATOW_3D, KOMPENSACJE_DEFINICJE, POLACZENIA_SZKIELETU
from kinematics_3d import AnalizatorRuchu3D
import patient_tracker as tracker


class KineTrackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KineTrack 3D — Analiza Nagrań Pacjentów, Nakładanie Szkieletu i Raporty")
        self.root.geometry("1360x820")

        self.analizator_live = None
        self.cap = None
        self.tryb = "STOP"  # "LIVE", "MP4", "STOP"
        self.nagrywanie_sesji = False

        self.pokaz_katy_stawow = tk.BooleanVar(value=True)
        self.pokaz_kompensacje_hud = tk.BooleanVar(value=True)
        self.zapisuj_wideo_z_szkieletem = tk.BooleanVar(value=True)

        self.historia_katow = []
        self.historia_komp = []
        self.ostatni_rekord_raportu = None
        self.start_czas_ms = 0

        self._zbuduj_ui()

    def _zbuduj_ui(self):
        # Górny pasek sterowania
        top_bar = tk.Frame(self.root, bg="#1e293b", pady=8, padx=10)
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text="Pacjent:", fg="white", bg="#1e293b", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.entry_pacjent = tk.Entry(top_bar, width=13, font=("Segoe UI", 9))
        self.entry_pacjent.insert(0, "Pacjent_01")
        self.entry_pacjent.pack(side=tk.LEFT, padx=(4, 10))

        tk.Label(top_bar, text="Ćwiczenie:", fg="white", bg="#1e293b", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.entry_cwiczenie = tk.Entry(top_bar, width=16, font=("Segoe UI", 9))
        self.entry_cwiczenie.insert(0, "Przysiad_Jednonoz")
        self.entry_cwiczenie.pack(side=tk.LEFT, padx=(4, 10))

        tk.Button(top_bar, text="📂 Wgraj nagranie pacjenta (MP4/AVI)", bg="#3b82f6", fg="white",
                  font=("Segoe UI", 9, "bold"), command=self.analizuj_plik_wideo).pack(side=tk.LEFT, padx=3)

        tk.Button(top_bar, text="📹 Kamera LIVE", bg="#10b981", fg="white",
                  font=("Segoe UI", 9, "bold"), command=self.start_kamera).pack(side=tk.LEFT, padx=3)

        self.btn_zapis_live = tk.Button(top_bar, text="⏺ Rejestruj LIVE", bg="#f59e0b", fg="black",
                                        font=("Segoe UI", 9, "bold"), command=self.przelacz_rejestracje_live, state=tk.DISABLED)
        self.btn_zapis_live.pack(side=tk.LEFT, padx=3)

        self.btn_zapisz_raport = tk.Button(top_bar, text="💾 Zapisz raport z ćwiczenia", bg="#ec4899", fg="white",
                                           font=("Segoe UI", 9, "bold"), command=self.zapisz_raport_recznie, state=tk.DISABLED)
        self.btn_zapisz_raport.pack(side=tk.LEFT, padx=(8, 3))

        tk.Button(top_bar, text="📈 Postępy", bg="#8b5cf6", fg="white",
                  font=("Segoe UI", 9, "bold"), command=self.pokaz_wykres_postepow).pack(side=tk.RIGHT, padx=3)

        # Drugi pasek: opcje nakładania szkieletu i eksportu wideo
        sub_bar = tk.Frame(self.root, bg="#0f172a", pady=4, padx=12)
        sub_bar.pack(fill=tk.X)

        tk.Checkbutton(sub_bar, text="Pokazuj kąty stawów 3D na szkielecie", variable=self.pokaz_katy_stawow,
                       bg="#0f172a", fg="#4ade80", selectcolor="#1e293b", activebackground="#0f172a",
                       font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 14))

        tk.Checkbutton(sub_bar, text="Pokazuj alerty kompensacji na szkielecie", variable=self.pokaz_kompensacje_hud,
                       bg="#0f172a", fg="#f87171", selectcolor="#1e293b", activebackground="#0f172a",
                       font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 14))

        tk.Checkbutton(sub_bar, text="Zapisz wideo MP4 z nałożonym szkieletem po analizie pliku", variable=self.zapisuj_wideo_z_szkieletem,
                       bg="#0f172a", fg="#38bdf8", selectcolor="#1e293b", activebackground="#0f172a",
                       font=("Segoe UI", 9)).pack(side=tk.LEFT)

        # Główny obszar dzielony
        content = tk.Frame(self.root, bg="#0f172a")
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        video_frame = tk.Frame(content, bg="black")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lbl_wideo = tk.Label(
            video_frame, bg="black",
            text="Wgraj surowe nagranie pacjenta (MP4 / AVI / MOV) lub uruchom kamerę na żywo.\nProgram automatycznie nałoży szkielet 3D, zmierzy kąty i wykryje kompensacje.",
            fg="#94a3b8", font=("Segoe UI", 11)
        )
        self.lbl_wideo.pack(fill=tk.BOTH, expand=True)

        # Prawy panel telemetryczny
        telemetry_frame = tk.Frame(content, width=420, bg="#1e293b", padx=10, pady=10)
        telemetry_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        telemetry_frame.pack_propagate(False)

        self.lbl_fps = tk.Label(telemetry_frame, text="Status: Gotowy do analizy", fg="#38bdf8", bg="#1e293b", font=("Segoe UI", 9, "bold"))
        self.lbl_fps.pack(anchor="w")

        tk.Label(telemetry_frame, text="DETEKCJA KOMPENSACJI (3D):", fg="#f87171", bg="#1e293b", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 4))
        self.tree_komp = ttk.Treeview(telemetry_frame, columns=("nazwa", "wartosc", "status"), show="headings", height=11)
        self.tree_komp.heading("nazwa", text="Wzorzec kompensacji")
        self.tree_komp.heading("wartosc", text="Kąt")
        self.tree_komp.heading("status", text="Stan")
        self.tree_komp.column("nazwa", width=250)
        self.tree_komp.column("wartosc", width=65, anchor="center")
        self.tree_komp.column("status", width=85, anchor="center")
        self.tree_komp.pack(fill=tk.X)

        for d in KOMPENSACJE_DEFINICJE:
            self.tree_komp.insert("", tk.END, iid=d["nazwa"], values=(d["nazwa"], "0.0°", "OK"))

        tk.Label(telemetry_frame, text="KĄTY STAWÓW 3D:", fg="#4ade80", bg="#1e293b", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))
        self.tree_katy = ttk.Treeview(telemetry_frame, columns=("staw", "kat"), show="headings", height=12)
        self.tree_katy.heading("staw", text="Staw / Segment")
        self.tree_katy.heading("kat", text="Kąt 3D")
        self.tree_katy.column("staw", width=305)
        self.tree_katy.column("kat", width=95, anchor="center")
        self.tree_katy.pack(fill=tk.BOTH, expand=True)

        for nazwa in NAZWY_KATOW_3D:
            self.tree_katy.insert("", tk.END, iid=nazwa, values=(nazwa, "--"))

    # --- RYSOWANIE SZKIELETU, ŁUKÓW KĄTOWYCH I KOMPENSACJI ---

    @staticmethod
    def _rysuj_etykiete_hud(img, tekst, pozycja, kolor_tekstu=(255, 255, 255), kolor_tla=(15, 23, 42), skala=0.52, grubosc=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(tekst, font, skala, grubosc)
        x, y = int(pozycja[0]), int(pozycja[1])

        h, w, _ = img.shape
        x = max(4, min(x, w - tw - 8))
        y = max(th + 6, min(y, h - 6))

        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2), kolor_tla, -1)
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2), kolor_tekstu, 1)
        cv2.putText(img, tekst, (x, y), font, skala, kolor_tekstu, grubosc, cv2.LINE_AA)

    @staticmethod
    def _rysuj_luk_kata(img, p_a, p_b, p_c, kolor=(74, 222, 128), promien=22):
        ang_a = math.degrees(math.atan2(p_a[1] - p_b[1], p_a[0] - p_b[0]))
        ang_c = math.degrees(math.atan2(p_c[1] - p_b[1], p_c[0] - p_b[0]))
        diff = (ang_c - ang_a + 360) % 360
        if diff > 180:
            ang_a, ang_c = ang_c, ang_a
            diff = 360 - diff
        cv2.ellipse(img, p_b, (promien, promien), 0, ang_a, ang_a + diff, kolor, 2, cv2.LINE_AA)

    @staticmethod
    def _rysuj_linie_przerywana(img, p1, p2, kolor=(250, 204, 21), grubosc=2, dlugosc_kreski=8):
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 1:
            return
        kroki = int(dist / dlugosc_kreski)
        for i in range(0, kroki, 2):
            t1 = i / max(kroki, 1)
            t2 = min((i + 1) / max(kroki, 1), 1.0)
            pt1 = (int(p1[0] + (p2[0] - p1[0]) * t1), int(p1[1] + (p2[1] - p1[1]) * t1))
            pt2 = (int(p1[0] + (p2[0] - p1[0]) * t2), int(p1[1] + (p2[1] - p1[1]) * t2))
            cv2.line(img, pt1, pt2, kolor, grubosc, cv2.LINE_AA)

    def _rysuj_hud_3d(self, rgb, p2d, katy, kompensacje):
        if p2d is None or katy is None or kompensacje is None:
            return
        h, w, _ = rgb.shape
        pts = {i: (int(p2d[i][0] * w), int(p2d[i][1] * h)) for i in range(len(p2d))}
        progi = [d["prog"] for d in KOMPENSACJE_DEFINICJE]

        alert_tulow = kompensacje[0] > progi[0] or kompensacje[1] > progi[1] or kompensacje[2] > progi[2]
        alert_barki = kompensacje[3] > progi[3] or kompensacje[4] > progi[4]
        alert_miednica = kompensacje[5] > progi[5] or kompensacje[6] > progi[6]
        alert_kolano_l = kompensacje[7] > progi[7] or kompensacje[9] > progi[9] or kompensacje[11] > progi[11]
        alert_kolano_p = kompensacje[8] > progi[8] or kompensacje[10] > progi[10] or kompensacje[12] > progi[12]

        # 1. Szkielet nałożony na surowy obraz pacjenta
        for a, b in POLACZENIA_SZKIELETU:
            if a in pts and b in pts:
                kolor = (56, 189, 248)
                grubosc = 3
                if (a, b) == (11, 12) and alert_barki:
                    kolor, grubosc = (239, 68, 68), 5
                elif (a, b) == (23, 24) and alert_miednica:
                    kolor, grubosc = (239, 68, 68), 5
                elif (a, b) in ((11, 23), (12, 24)) and alert_tulow:
                    kolor, grubosc = (249, 115, 22), 5
                elif (a, b) in ((23, 25), (25, 27)) and alert_kolano_l:
                    kolor, grubosc = (239, 68, 68), 5
                elif (a, b) in ((24, 26), (26, 28)) and alert_kolano_p:
                    kolor, grubosc = (239, 68, 68), 5

                cv2.line(rgb, pts[a], pts[b], kolor, grubosc, cv2.LINE_AA)

        # 2. Kąty stawowe bezpośrednio przy stawach
        if self.pokaz_katy_stawow.get():
            stawy_do_rysowania = [
                (0, 23, 11, 13, (12, -8)),
                (1, 24, 12, 14, (-68, -8)),
                (2, 11, 13, 15, (12, 0)),
                (3, 12, 14, 16, (-68, 0)),
                (6, 11, 23, 25, (14, 5)),
                (7, 12, 24, 26, (-70, 5)),
                (8, 23, 25, 27, (14, 0)),
                (9, 24, 26, 28, (-70, 0)),
                (10, 25, 27, 31, (12, 12)),
                (11, 26, 28, 32, (-68, 12)),
            ]
            for idx_k, a, b, c, (dx, dy) in stawy_do_rysowania:
                if all(k in pts for k in (a, b, c)):
                    val = katy[idx_k]
                    self._rysuj_luk_kata(rgb, pts[a], pts[b], pts[c], kolor=(74, 222, 128), promien=20)
                    self._rysuj_etykiete_hud(
                        rgb, f"{val:.0f} st.", (pts[b][0] + dx, pts[b][1] + dy),
                        kolor_tekstu=(74, 222, 128), kolor_tla=(15, 23, 42), skala=0.48
                    )

        # 3. Alerty kompensacji w miejscu wystąpienia
        if self.pokaz_kompensacje_hud.get():
            if alert_tulow and all(k in pts for k in (11, 12, 23, 24)):
                s_barki = ((pts[11][0] + pts[12][0]) // 2, (pts[11][1] + pts[12][1]) // 2)
                s_biodra = ((pts[23][0] + pts[24][0]) // 2, (pts[23][1] + pts[24][1]) // 2)
                pion_odniesienia = (s_biodra[0], s_barki[1])
                self._rysuj_linie_przerywana(rgb, s_biodra, pion_odniesienia, kolor=(250, 204, 21), grubosc=2)
                cv2.line(rgb, s_biodra, s_barki, (249, 115, 22), 3, cv2.LINE_AA)
                naj_tulow = max(
                    [(kompensacje[0], "POCHYLENIE BOK"), (kompensacje[1], "POCHYLENIE PRZOD/TYL"), (kompensacje[2], "ROTACJA TULOWIA")],
                    key=lambda x: x[0]
                )
                self._rysuj_etykiete_hud(
                    rgb, f"! {naj_tulow[1]}: {naj_tulow[0]:.1f} st.",
                    (s_barki[0] - 80, s_barki[1] - 28),
                    kolor_tekstu=(255, 255, 255), kolor_tla=(185, 28, 28), skala=0.52
                )

            if alert_barki and 11 in pts and 12 in pts:
                y_poziom = max(pts[11][1], pts[12][1])
                self._rysuj_linie_przerywana(rgb, (pts[11][0], y_poziom), (pts[12][0], y_poziom), kolor=(250, 204, 21), grubosc=2)
                wyzszy_bark = pts[11] if pts[11][1] < pts[12][1] else pts[12]
                self._rysuj_etykiete_hud(
                    rgb, f"! UNIESIENIE BARKU: {kompensacje[3]:.1f} st.",
                    (wyzszy_bark[0] - 60, wyzszy_bark[1] - 22),
                    kolor_tekstu=(255, 255, 255), kolor_tla=(185, 28, 28), skala=0.50
                )

            if alert_miednica and 23 in pts and 24 in pts:
                y_miednica = max(pts[23][1], pts[24][1])
                self._rysuj_linie_przerywana(rgb, (pts[23][0], y_miednica), (pts[24][0], y_miednica), kolor=(250, 204, 21), grubosc=2)
                s_miednica = ((pts[23][0] + pts[24][0]) // 2, (pts[23][1] + pts[24][1]) // 2)
                self._rysuj_etykiete_hud(
                    rgb, f"! MIEDNICA: {kompensacje[5]:.1f} st.",
                    (s_miednica[0] - 65, s_miednica[1] + 22),
                    kolor_tekstu=(255, 255, 255), kolor_tla=(185, 28, 28), skala=0.50
                )

            if alert_kolano_l and all(k in pts for k in (23, 25, 27)):
                self._rysuj_linie_przerywana(rgb, pts[23], pts[27], kolor=(250, 204, 21), grubosc=2)
                typ_l = "KOSLAWIENIE" if kompensacje[7] > progi[7] else ("SZPOTAWIENIE" if kompensacje[9] > progi[9] else "PRZEPROST")
                val_l = max(kompensacje[7], kompensacje[9], kompensacje[11])
                self._rysuj_etykiete_hud(
                    rgb, f"! {typ_l} L: {val_l:.1f} st.",
                    (pts[25][0] + 16, pts[25][1] + 24),
                    kolor_tekstu=(255, 255, 255), kolor_tla=(185, 28, 28), skala=0.50
                )

            if alert_kolano_p and all(k in pts for k in (24, 26, 28)):
                self._rysuj_linie_przerywana(rgb, pts[24], pts[28], kolor=(250, 204, 21), grubosc=2)
                typ_p = "KOSLAWIENIE" if kompensacje[8] > progi[8] else ("SZPOTAWIENIE" if kompensacje[10] > progi[10] else "PRZEPROST")
                val_p = max(kompensacje[8], kompensacje[10], kompensacje[12])
                self._rysuj_etykiete_hud(
                    rgb, f"! {typ_p} P: {val_p:.1f} st.",
                    (pts[26][0] - 155, pts[26][1] + 24),
                    kolor_tekstu=(255, 255, 255), kolor_tla=(185, 28, 28), skala=0.50
                )

        for idx in range(11, 33):
            if idx in pts:
                c = (239, 68, 68) if ((idx == 25 and alert_kolano_l) or (idx == 26 and alert_kolano_p)) else (74, 222, 128)
                cv2.circle(rgb, pts[idx], 6, c, -1, cv2.LINE_AA)

    def _aktualizuj_tabele(self, katy, kompensacje):
        if katy is None or kompensacje is None:
            return
        for idx, nazwa in enumerate(NAZWY_KATOW_3D):
            self.tree_katy.item(nazwa, values=(nazwa, f"{katy[idx]:.1f}°"))

        for idx, definicja in enumerate(KOMPENSACJE_DEFINICJE):
            nazwa = definicja["nazwa"]
            val = kompensacje[idx]
            przekr = val > definicja["prog"]
            status = "⚠ KOMPENSACJA" if przekr else "OK"
            self.tree_komp.item(nazwa, values=(nazwa, f"{val:.1f}°", status))

    # --- WGRYWANIE SUROWEGO PLIKU WIDEO PACJENTA I NAKŁADANIE SZKIELETU ---

    def analizuj_plik_wideo(self):
        sciezka = filedialog.askopenfilename(
            title="Wybierz nagranie pacjenta (bez szkieletu)",
            filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        if not sciezka:
            return

        self._zatrzymaj_zrodlo()
        self.tryb = "MP4"
        self.btn_zapis_live.config(state=tk.DISABLED)
        self.btn_zapisz_raport.config(state=tk.DISABLED)
        threading.Thread(target=self._watek_analizy_wideo, args=(sciezka,), daemon=True).start()

    def _watek_analizy_wideo(self, sciezka):
        try:
            lokalny_analizator = AnalizatorRuchu3D()

            cap = cv2.VideoCapture(sciezka)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Błąd pliku", "OpenCV nie mogło otworzyć wybranego pliku wideo."))
                self.tryb = "STOP"
                return

            fps_wideo = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            wszystkie_klatki = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            id_pacjenta = "".join(c for c in self.entry_pacjent.get().strip() if c.isalnum() or c in ("_", "-")) or "Pacjent_01"
            cwiczenie = "".join(c for c in self.entry_cwiczenie.get().strip() if c.isalnum() or c in ("_", "-")) or "Test_Ruchowy"

            sciezka_wideo_hud = ""
            writer = None
            if self.zapisuj_wideo_z_szkieletem.get() and w > 0 and h > 0:
                folder_pacjenta = os.path.join(FOLDER_PACJENTOW, id_pacjenta)
                os.makedirs(folder_pacjenta, exist_ok=True)
                znacznik = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                sciezka_wideo_hud = os.path.join(folder_pacjenta, f"{znacznik}_{cwiczenie}_SZKIELET_3D.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(sciezka_wideo_hud, fourcc, fps_wideo, (w, h))

            self.historia_katow.clear()
            self.historia_komp.clear()

            nr = 0
            t_start = time.perf_counter()

            while self.tryb == "MP4":
                ret, frame = cap.read()
                if not ret:
                    break
                nr += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = int((nr / fps_wideo) * 1000.0)

                p2d, katy, komp = lokalny_analizator.przetworz_klatke(rgb, ts_ms)
                if katy is not None:
                    self.historia_katow.append(katy)
                    self.historia_komp.append(komp)

                self._rysuj_hud_3d(rgb, p2d, katy, komp)

                if writer is not None:
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                if nr % 2 == 0 or nr == wszystkie_klatki:
                    fps_analizy = nr / max(time.perf_counter() - t_start, 1e-4)
                    self.root.after(0, self._odswiez_ui_mp4, rgb.copy(), katy, komp, nr, wszystkie_klatki, fps_analizy)

            cap.release()
            if writer is not None:
                writer.release()

            self.tryb = "STOP"
            nazwa_pliku = os.path.basename(sciezka)
            self.root.after(0, lambda: self._zakoncz_analize_i_przygotuj_raport(f"Nagranie: {nazwa_pliku}", sciezka_wideo_hud))

        except Exception as e:
            self.tryb = "STOP"
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Błąd podczas analizy", f"Wystąpił wyjątek:\n{err_msg}"))

    def _odswiez_ui_mp4(self, rgb, katy, komp, nr, total, fps):
        self.lbl_fps.config(text=f"Nakładanie szkieletu: klatka {nr}/{total} ({fps:.1f} FPS)")
        self._aktualizuj_tabele(katy, komp)
        self._wyswietl_obraz(rgb)

    # --- TRYB KAMERY NA ŻYWO ---

    def start_kamera(self):
        self._zatrzymaj_zrodlo()
        if self.analizator_live is None:
            self.analizator_live = AnalizatorRuchu3D()
        else:
            self.analizator_live.resetuj_sesje()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.tryb = "LIVE"
        self.nagrywanie_sesji = False
        self.btn_zapis_live.config(state=tk.NORMAL, text="⏺ Rejestruj LIVE", bg="#f59e0b")
        self.start_czas_ms = int(time.time() * 1000)
        self._petla_kamery()

    def przelacz_rejestracje_live(self):
        if not self.nagrywanie_sesji:
            self.historia_katow.clear()
            self.historia_komp.clear()
            self.nagrywanie_sesji = True
            self.btn_zapis_live.config(text="⏹ Zakończ próbę", bg="#ef4444", fg="white")
        else:
            self.nagrywanie_sesji = False
            self.btn_zapis_live.config(text="⏺ Rejestruj LIVE", bg="#f59e0b", fg="black")
            self._zakoncz_analize_i_przygotuj_raport("Kamera_LIVE", "")

    def _petla_kamery(self):
        if self.tryb != "LIVE" or self.cap is None:
            return

        t0 = time.perf_counter()
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int(time.time() * 1000) - self.start_czas_ms

            p2d, katy, komp = self.analizator_live.przetworz_klatke(rgb, ts)
            self._rysuj_hud_3d(rgb, p2d, katy, komp)
            self._aktualizuj_tabele(katy, komp)

            if self.nagrywanie_sesji and katy is not None:
                self.historia_katow.append(katy)
                self.historia_komp.append(komp)
                cv2.circle(rgb, (30, 30), 12, (239, 68, 68), -1)

            fps = 1.0 / max(time.perf_counter() - t0, 1e-4)
            self.lbl_fps.config(text=f"Wydajność: {fps:.1f} FPS | Tryb: LIVE 3D")
            self._wyswietl_obraz(rgb)

        self.root.after(10, self._petla_kamery)

    # --- RAPORTOWANIE I ZAPIS WYNIKÓW ĆWICZENIA ---

    def _zakoncz_analize_i_przygotuj_raport(self, zrodlo, sciezka_wideo_hud):
        wynik = tracker.analizuj_pelna_sekwencje(self.historia_katow, self.historia_komp)
        if not wynik:
            messagebox.showwarning("Brak detekcji", "Nie udało się wykryć sylwetki pacjenta na nagraniu.")
            return

        id_pacjenta = self.entry_pacjent.get().strip() or "Pacjent_01"
        cwiczenie = self.entry_cwiczenie.get().strip() or "Test_Ruchowy"

        rekord = tracker.utworz_rekord_raportu(id_pacjenta, cwiczenie, zrodlo, wynik, sciezka_wideo_hud)
        tracker.zapisz_wynik_pacjenta(rekord)
        self.ostatni_rekord_raportu = rekord
        self.btn_zapisz_raport.config(state=tk.NORMAL)
        self.lbl_fps.config(text=f"Gotowe! MQI: {rekord['mqi']}% | Zapisano w bazie pacjenta")

        komp_lista = "\n".join(
            [f"• {k['nazwa']}: {k['szczyt']}° (norma ≤ {k['prog']}°, +{k['przekroczenie']}°)"
             for k in wynik["wykryte_kompensacje"]]
        ) or "• Brak istotnych kompensacji — ruch czysty biomechanicznie!"

        rom_lista = "\n".join(
            [f"• {s['staw']}: {s['min']}° – {s['max']}° (ROM: {s['rom']}°)"
             for s in wynik["stawy_rom"][:8]]
        )

        info_wideo = f"\nWideo z nałożonym szkieletem zapisano jako:\n{os.path.basename(sciezka_wideo_hud)}\n" if sciezka_wideo_hud else ""

        odp = messagebox.askyesno(
            f"Analiza zakończona — {id_pacjenta}",
            f"Ćwiczenie: {cwiczenie}\n"
            f"Indeks Jakości Ruchu (MQI): {wynik['mqi']} / 100%\n"
            f"{info_wideo}\n"
            f"WYKRYTE KOMPENSACJE:\n{komp_lista}\n\n"
            f"ZAKRESY RUCHU STAWÓW (ROM 3D):\n{rom_lista}\n\n"
            f"Czy chcesz teraz zapisać raport z ćwiczenia do pliku (HTML / TXT)?"
        )
        if odp:
            self.zapisz_raport_recznie()

    def zapisz_raport_recznie(self):
        if not self.ostatni_rekord_raportu:
            messagebox.showinfo("Brak danych", "Najpierw przeanalizuj nagranie pacjenta lub próbę na żywo.")
            return

        r = self.ostatni_rekord_raportu
        domyslna_nazwa = f"Raport_{r['pacjent']}_{r['cwiczenie']}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.html"
        sciezka = filedialog.asksaveasfilename(
            title="Zapisz raport z wykonania ćwiczenia",
            initialfile=domyslna_nazwa,
            defaultextension=".html",
            filetypes=[("Raport HTML (do druku / przeglądarki)", "*.html"), ("Raport tekstowy TXT", "*.txt")]
        )
        if not sciezka:
            return

        tracker.eksportuj_raport_tekstowy(r, sciezka)
        messagebox.showinfo("Raport zapisany", f"Raport z wykonania ćwiczenia został zapisany:\n{sciezka}")

    def pokaz_wykres_postepow(self):
        id_pacjenta = self.entry_pacjent.get().strip() or "Pacjent_01"
        historia = tracker.pobierz_historie_pacjenta(id_pacjenta)
        if not historia:
            messagebox.showinfo("Brak historii", f"Brak zapisanych badań dla pacjenta '{id_pacjenta}'.")
            return

        daty = [f"{h['data']}\n({h['cwiczenie']})" for h in historia]
        mqi = [h["mqi"] for h in historia]
        liczba_komp = [len(h["wykryte_kompensacje"]) for h in historia]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(daty, mqi, marker="o", color="#10b981", linewidth=2.5)
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Indeks Jakości Ruchu MQI (%)")
        ax1.set_title(f"Monitorowanie postępów rehabilitacji: {id_pacjenta}")
        ax1.grid(True, alpha=0.3)

        ax2.bar(daty, liczba_komp, color="#ef4444", alpha=0.8)
        ax2.set_ylabel("Liczba wykrytych kompensacji")
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=15, fontsize=8)
        plt.tight_layout()
        plt.show()

    def _wyswietl_obraz(self, rgb):
        h, w, _ = rgb.shape
        skala = min(840 / max(w, 1), 680 / max(h, 1))
        nowy_w, nowy_h = int(w * skala), int(h * skala)
        resized = cv2.resize(rgb, (nowy_w, nowy_h), interpolation=cv2.INTER_LINEAR)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.lbl_wideo.imgtk = imgtk
        self.lbl_wideo.configure(image=imgtk)

    def _zatrzymaj_zrodlo(self):
        self.tryb = "STOP"
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def zamknij(self):
        self._zatrzymaj_zrodlo()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = KineTrackApp(root)
    root.protocol("WM_DELETE_WINDOW", app.zamknij)
    root.mainloop()