import os
import time
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from biomechanics_config import NAZWY_KATOW_3D, KOMPENSACJE_DEFINICJE, POLACZENIA_SZKIELETU
from kinematics_3d import AnalizatorRuchu3D
import patient_tracker as tracker


class KineTrackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KineTrack 3D — Analiza Biomechaniczna i Wizualizacja Kątów na Szkielecie")
        self.root.geometry("1320x800")

        self.analizator = AnalizatorRuchu3D()
        self.cap = None
        self.tryb = "STOP"  # "LIVE", "MP4", "STOP"
        self.nagrywanie_sesji = False

        # Przełączniki warstw HUD na szkielecie
        self.pokaz_katy_stawow = tk.BooleanVar(value=True)
        self.pokaz_kompensacje_hud = tk.BooleanVar(value=True)

        self.historia_katow = []
        self.historia_komp = []
        self.start_czas_ms = 0

        self._zbuduj_ui()

    def _zbuduj_ui(self):
        # Górny pasek sterowania
        top_bar = tk.Frame(self.root, bg="#1e293b", pady=8, padx=12)
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text="ID Pacjenta:", fg="white", bg="#1e293b", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.entry_pacjent = tk.Entry(top_bar, width=15, font=("Segoe UI", 10))
        self.entry_pacjent.insert(0, "Pacjent_01")
        self.entry_pacjent.pack(side=tk.LEFT, padx=(5, 12))

        tk.Label(top_bar, text="Ćwiczenie:", fg="white", bg="#1e293b", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.entry_cwiczenie = tk.Entry(top_bar, width=18, font=("Segoe UI", 10))
        self.entry_cwiczenie.insert(0, "Przysiad_Jednonoz")
        self.entry_cwiczenie.pack(side=tk.LEFT, padx=(5, 12))

        tk.Button(top_bar, text="📹 Kamera na żywo", bg="#10b981", fg="white", font=("Segoe UI", 9, "bold"),
                  command=self.start_kamera).pack(side=tk.LEFT, padx=3)
        self.btn_zapis_live = tk.Button(top_bar, text="⏺ Rejestruj (LIVE)", bg="#f59e0b", fg="black",
                                        font=("Segoe UI", 9, "bold"), command=self.przelacz_rejestracje_live, state=tk.DISABLED)
        self.btn_zapis_live.pack(side=tk.LEFT, padx=3)

        tk.Button(top_bar, text="📂 Analizuj MP4", bg="#3b82f6", fg="white", font=("Segoe UI", 9, "bold"),
                  command=self.analizuj_plik_mp4).pack(side=tk.LEFT, padx=3)

        # Przełączniki widoczności na szkielecie
        tk.Checkbutton(top_bar, text="Kąty stawów na ciele", variable=self.pokaz_katy_stawow,
                       bg="#1e293b", fg="#4ade80", selectcolor="#0f172a", activebackground="#1e293b",
                       font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        tk.Checkbutton(top_bar, text="Alerty kompensacji na ciele", variable=self.pokaz_kompensacje_hud,
                       bg="#1e293b", fg="#f87171", selectcolor="#0f172a", activebackground="#1e293b",
                       font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)

        tk.Button(top_bar, text="📈 Postępy", bg="#8b5cf6", fg="white", font=("Segoe UI", 9, "bold"),
                  command=self.pokaz_wykres_postepow).pack(side=tk.RIGHT, padx=3)

        # Główny obszar dzielony
        content = tk.Frame(self.root, bg="#0f172a")
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        video_frame = tk.Frame(content, bg="black")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lbl_wideo = tk.Label(video_frame, bg="black", text="Wybierz kamerę na żywo lub wczytaj plik MP4", fg="#94a3b8", font=("Segoe UI", 12))
        self.lbl_wideo.pack(fill=tk.BOTH, expand=True)

        # Prawy panel telemetryczny
        telemetry_frame = tk.Frame(content, width=420, bg="#1e293b", padx=10, pady=10)
        telemetry_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        telemetry_frame.pack_propagate(False)

        self.lbl_fps = tk.Label(telemetry_frame, text="Wydajność: -- FPS | Status: Gotowy", fg="#38bdf8", bg="#1e293b", font=("Segoe UI", 9, "bold"))
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

    # --- POMOCNICZE RYSOWANIE NA OBRAZIE ---

    @staticmethod
    def _rysuj_etykiete_hud(img, tekst, pozycja, kolor_tekstu=(255, 255, 255), kolor_tla=(15, 23, 42), skala=0.52, grubosc=1):
        """Rysuje czytelną etykietę z ciemnym tłem i obramowaniem bezpośrednio przy stawie."""
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
    def _rysuj_luk_kata(img, p_a, p_b, p_c, kolor=(74, 222, 128), promien=24):
        """Rysuje wycinek koła (łuk) między dwoma segmentami w stawie p_b."""
        ang_a = math.degrees(math.atan2(p_a[1] - p_b[1], p_a[0] - p_b[0]))
        ang_c = math.degrees(math.atan2(p_c[1] - p_b[1], p_c[0] - p_b[0]))

        diff = (ang_c - ang_a + 360) % 360
        if diff > 180:
            ang_a, ang_c = ang_c, ang_a
            diff = 360 - diff

        cv2.ellipse(img, p_b, (promien, promien), 0, ang_a, ang_a + diff, kolor, 2, cv2.LINE_AA)

    @staticmethod
    def _rysuj_linie_przerywana(img, p1, p2, kolor=(250, 204, 21), grubosc=2, dlugosc_kreski=8):
        """Rysuje przerywaną linię odniesienia (pion / oś prawidłowa)."""
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

    # --- GŁÓWNE RYSOWANIE SZKIELETU + KĄTÓW + KOMPENSACJI ---

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

        # 1. Rysowanie kości szkieletu
        for a, b in POLACZENIA_SZKIELETU:
            if a in pts and b in pts:
                kolor = (56, 189, 248)
                grubosc = 2
                if (a, b) == (11, 12) and alert_barki:
                    kolor, grubosc = (239, 68, 68), 4
                elif (a, b) == (23, 24) and alert_miednica:
                    kolor, grubosc = (239, 68, 68), 4
                elif (a, b) in ((11, 23), (12, 24)) and alert_tulow:
                    kolor, grubosc = (249, 115, 22), 4
                elif (a, b) in ((23, 25), (25, 27)) and alert_kolano_l:
                    kolor, grubosc = (239, 68, 68), 4
                elif (a, b) in ((24, 26), (26, 28)) and alert_kolano_p:
                    kolor, grubosc = (239, 68, 68), 4

                cv2.line(rgb, pts[a], pts[b], kolor, grubosc, cv2.LINE_AA)

        # 2. Rysowanie łuków i wartości kątów 3D bezpośrednio przy stawach
        if self.pokaz_katy_stawow.get():
            stawy_do_rysowania = [
                (0, 23, 11, 13, (12, -8)),     # Bark lewy
                (1, 24, 12, 14, (-68, -8)),    # Bark prawy
                (2, 11, 13, 15, (12, 0)),      # Łokieć lewy
                (3, 12, 14, 16, (-68, 0)),     # Łokieć prawy
                (6, 11, 23, 25, (14, 5)),      # Biodro lewe
                (7, 12, 24, 26, (-70, 5)),     # Biodro prawe
                (8, 23, 25, 27, (14, 0)),      # Kolano lewe
                (9, 24, 26, 28, (-70, 0)),     # Kolano prawe
                (10, 25, 27, 31, (12, 12)),    # Kostka lewa
                (11, 26, 28, 32, (-68, 12)),   # Kostka prawa
            ]

            for idx_k, a, b, c, (dx, dy) in stawy_do_rysowania:
                if all(k in pts for k in (a, b, c)):
                    val = katy[idx_k]
                    self._rysuj_luk_kata(rgb, pts[a], pts[b], pts[c], kolor=(74, 222, 128), promien=20)
                    self._rysuj_etykiete_hud(
                        rgb, f"{val:.0f} st.", (pts[b][0] + dx, pts[b][1] + dy),
                        kolor_tekstu=(74, 222, 128), kolor_tla=(15, 23, 42), skala=0.48
                    )

        # 3. Bezpośrednie rysowanie linii odniesienia i etykiet kompensacji w miejscu błędu
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
                    kolor_tekstu=(255, 255, 255), kolor_tla=(185, 28, 28), skala=0.52, grubosc=1
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

        # 4. Punkty stawowe na wierzchu
        for idx in range(11, 33):
            if idx in pts:
                c = (239, 68, 68) if ((idx == 25 and alert_kolano_l) or (idx == 26 and alert_kolano_p)) else (74, 222, 128)
                cv2.circle(rgb, pts[idx], 5, c, -1, cv2.LINE_AA)

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

    # --- TRYB KAMERY NA ŻYWO ---

    def start_kamera(self):
        self._zatrzymaj_zrodlo()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.analizator.resetuj_sesje()
        self.tryb = "LIVE"
        self.nagrywanie_sesji = False
        self.btn_zapis_live.config(state=tk.NORMAL, text="⏺ Rejestruj (LIVE)", bg="#f59e0b")
        self.start_czas_ms = int(time.time() * 1000)
        self._petla_kamery()

    def przelacz_rejestracje_live(self):
        if not self.nagrywanie_sesji:
            self.historia_katow.clear()
            self.historia_komp.clear()
            self.nagrywanie_sesji = True
            self.btn_zapis_live.config(text="⏹ Zakończ i oceń", bg="#ef4444", fg="white")
        else:
            self.nagrywanie_sesji = False
            self.btn_zapis_live.config(text="⏺ Rejestruj (LIVE)", bg="#f59e0b", fg="black")
            self._podsumuj_i_zapisz("Kamera_LIVE")

    def _petla_kamery(self):
        if self.tryb != "LIVE" or self.cap is None:
            return

        t0 = time.perf_counter()
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int(time.time() * 1000) - self.start_czas_ms

            p2d, katy, komp = self.analizator.przetworz_klatke(rgb, ts)
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

    # --- TRYB ANALIZY PLIKU MP4 ---

    def analizuj_plik_mp4(self):
        sciezka = filedialog.askopenfilename(filetypes=[("Wideo MP4/AVI/MOV", "*.mp4 *.avi *.mov *.mkv")])
        if not sciezka:
            return

        self._zatrzymaj_zrodlo()
        self.tryb = "MP4"
        self.btn_zapis_live.config(state=tk.DISABLED)
        threading.Thread(target=self._watek_mp4, args=(sciezka,), daemon=True).start()

    def _watek_mp4(self, sciezka):
        cap = cv2.VideoCapture(sciezka)
        fps_wideo = cap.get(cv2.CAP_PROP_FPS) or 30.0
        wszystkie_klatki = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.analizator.resetuj_sesje()
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

            p2d, katy, komp = self.analizator.przetworz_klatke(rgb, ts_ms)
            if katy is not None:
                self.historia_katow.append(katy)
                self.historia_komp.append(komp)

            if nr % 2 == 0 or nr == wszystkie_klatki:
                self._rysuj_hud_3d(rgb, p2d, katy, komp)
                fps_analizy = nr / max(time.perf_counter() - t_start, 1e-4)
                self.root.after(0, self._odswiez_ui_mp4, rgb, katy, komp, nr, wszystkie_klatki, fps_analizy)

        cap.release()
        self.tryb = "STOP"
        nazwa_pliku = os.path.basename(sciezka)
        self.root.after(0, lambda: self._podsumuj_i_zapisz(f"Plik: {nazwa_pliku}"))

    def _odswiez_ui_mp4(self, rgb, katy, komp, nr, total, fps):
        self.lbl_fps.config(text=f"Analiza MP4: klatka {nr}/{total} ({fps:.1f} FPS)")
        self._aktualizuj_tabele(katy, komp)
        self._wyswietl_obraz(rgb)

    # --- RAPORT I POSTĘPY ---

    def _podsumuj_i_zapisz(self, zrodlo):
        wynik = tracker.analizuj_pelna_sekwencje(self.historia_katow, self.historia_komp)
        if not wynik:
            messagebox.showwarning("Brak danych", "Nie wykryto sylwetki pacjenta na nagraniu.")
            return

        id_pacjenta = self.entry_pacjent.get().strip() or "Pacjent_01"
        cwiczenie = self.entry_cwiczenie.get().strip() or "Test_Ruchowy"
        tracker.zapisz_wynik_pacjenta(id_pacjenta, cwiczenie, zrodlo, wynik)

        komp_lista = "\n".join(
            [f"• {k['nazwa']}: {k['szczyt']}° (norma ≤ {k['prog']}°, +{k['przekroczenie']}°)"
             for k in wynik["wykryte_kompensacje"]]
        ) or "• Brak istotnych kompensacji — ruch czysty biomechanicznie!"

        rom_lista = "\n".join(
            [f"• {s['staw']}: {s['min']}° – {s['max']}° (zakres ROM: {s['rom']}°)"
             for s in wynik["stawy_rom"][:8]]
        )

        messagebox.showinfo(
            f"Raport Biomechaniczny 3D — {id_pacjenta}",
            f"Ćwiczenie: {cwiczenie}\n"
            f"Indeks Jakości Ruchu (MQI): {wynik['mqi']} / 100%\n"
            f"Przeanalizowane klatki: {wynik['liczba_klatek']}\n\n"
            f"WYKRYTE KOMPENSACJE (90. percentyl):\n{komp_lista}\n\n"
            f"ZAKRESY RUCHU STAWÓW (wybrane ROM 3D):\n{rom_lista}"
        )

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
