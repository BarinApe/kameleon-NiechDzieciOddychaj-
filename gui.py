import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import (
    DOMYSLNE_FPS, CZAS_NAGRANIA_IDEALU,
    MIN_KLATEK_PRZED_STOPEM, PROG_ZGODNOSCI_KONCA, WYMAGANE_KLATKI_KONCA,
    NAZWY_CECH, PRESETY_WAG
)
from pose_engine import SilnikSzkieletu, rysuj_tekst_z_obrysem
import storage_and_eval as db


class OknoWagStawow(tk.Toplevel):
    """Okno dialogowe pozwalające dobrać priorytety (wagi) stawów dla wybranego ćwiczenia."""
    def __init__(self, parent, nazwa_cwiczenia):
        super().__init__(parent)
        self.title(f"Wagi stawów: {nazwa_cwiczenia}")
        self.nazwa_cwiczenia = nazwa_cwiczenia
        self.suwaki = []

        tk.Label(
            self, text=f"Ustaw znaczenie stawów dla ćwiczenia '{nazwa_cwiczenia}':",
            font=("Arial", 10, "bold")
        ).pack(padx=10, pady=(10, 5))

        # Wybór gotowego szablonu
        preset_frame = tk.Frame(self)
        preset_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Label(preset_frame, text="Szablon:").pack(side=tk.LEFT)

        self.combo_preset = ttk.Combobox(preset_frame, values=list(PRESETY_WAG.keys()), state="readonly", width=35)
        self.combo_preset.pack(side=tk.LEFT, padx=5)
        self.combo_preset.bind("<<ComboboxSelected>>", self._zastosuj_preset)

        # Siatka 2 kolumn ze suwakami dla 16 stawów
        grid_frame = tk.Frame(self)
        grid_frame.pack(padx=10, pady=5)

        aktualne_wagi = db.wczytaj_wagi_cwiczenia(nazwa_cwiczenia)
        for idx, nazwa_cechy in enumerate(NAZWY_CECH):
            wiersz = idx % 8
            kolumna = (idx // 8) * 2

            tk.Label(grid_frame, text=nazwa_cechy, width=18, anchor="w").grid(row=wiersz, column=kolumna, padx=4, pady=2)
            suwak = tk.Scale(grid_frame, from_=0.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, length=130)
            suwak.set(aktualne_wagi[idx])
            suwak.grid(row=wiersz, column=kolumna + 1, padx=4, pady=2)
            self.suwaki.append(suwak)

        tk.Button(
            self, text="✔ Zapisz wagi dla tego ćwiczenia", bg="#c8e6c9",
            font=("Arial", 10, "bold"), command=self._zapisz
        ).pack(pady=10)

    def _zastosuj_preset(self, _event=None):
        wybrany = self.combo_preset.get()
        if wybrany in PRESETY_WAG:
            for suwak, wartosc in zip(self.suwaki, PRESETY_WAG[wybrany]):
                suwak.set(wartosc)

    def _zapisz(self):
        nowe_wagi = [float(s.get()) for s in self.suwaki]
        db.zapisz_wagi_cwiczenia(self.nazwa_cwiczenia, nowe_wagi)
        self.destroy()


class AplikacjaFizjo:
    def __init__(self, root):
        self.root = root
        self.root.title("System Analizy Ruchu Pacjenta (3D + Wagi Stawów + Raporty PDF)")
        self.silnik = SilnikSzkieletu()

        self.tryb_nagrywania = None        # None, 'ideal', 'sesja', 'analiza_pliku'
        self.odliczanie = 0
        self.pozostaly_czas_idealu = 0
        self.nazwa_na_ekranie = ""

        self.timer_start_id = None
        self.timer_ideal_id = None

        self.aktualna_sekwencja = []
        self.klatki_wideo = []
        self.lista_cwiczen = []

        self.indeks_sesji = 0
        self.wyniki_sesji = []
        self.docelowa_pozycja_koncowa = None
        self.licznik_klatek_koncowych = 0

        self.cap = cv2.VideoCapture(0)
        self._zbuduj_interfejs()
        self.odswiez_liste_cwiczen()
        self.aktualizuj_klatke()

    def _zbuduj_interfejs(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Lewy panel
        left_panel = tk.Frame(main_frame, width=290)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        tk.Label(left_panel, text="Lista ćwiczeń (kolejność):", font=("Arial", 11, "bold")).pack(anchor="w")

        self.listbox_cwiczenia = tk.Listbox(left_panel, width=36, height=14, font=("Arial", 10))
        self.listbox_cwiczenia.pack(pady=5, fill=tk.Y, expand=True)

        tk.Button(left_panel, text="+ Dodaj nowe ćwiczenie", bg="#e1f5fe", command=self.dodaj_cwiczenie).pack(fill=tk.X, pady=2)

        self.btn_ideal = tk.Button(
            left_panel, text=f"Nagraj ideał z kamery ({CZAS_NAGRANIA_IDEALU}s)",
            bg="#d0e8ff", command=self.przelacz_nagrywanie_idealu
        )
        self.btn_ideal.pack(fill=tk.X, pady=2)

        tk.Button(left_panel, text="⚖️ Ustaw wagi stawów dla ćwiczenia", bg="#fff9c4", command=self.otworz_okno_wag).pack(fill=tk.X, pady=2)

        tk.Label(left_panel, text="Pliki wideo i raporty:", font=("Arial", 9, "bold"), fg="#37474f").pack(anchor="w", pady=(8, 2))
        tk.Button(left_panel, text="📂 Wczytaj wideo jako ideał", bg="#e8eaf6", command=self.wczytaj_wideo_jako_ideal).pack(fill=tk.X, pady=2)
        tk.Button(left_panel, text="📊 Analizuj wideo pacjenta (próbka)", bg="#f3e5f5", command=self.analizuj_wideo_pacjenta).pack(fill=tk.X, pady=2)
        tk.Button(left_panel, text="📄 Generuj raport postępów (PDF)", bg="#e0f2f1", command=self.generuj_pdf_dla_zaznaczonego).pack(fill=tk.X, pady=2)

        tk.Button(left_panel, text="Usuń zaznaczone ćwiczenie", bg="#ffebee", command=self.usun_cwiczenie).pack(fill=tk.X, pady=(8, 2))

        # Prawy panel
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(right_panel)
        self.video_label.pack()

        self.status_label = tk.Label(
            right_panel, text="Status: Wybierz ćwiczenie lub rozpocznij sesję",
            font=("Arial", 12, "bold"), fg="#1a237e"
        )
        self.status_label.pack(pady=8)

        self.btn_sesja = tk.Button(
            right_panel, text="▶ Rozpocznij próbę pacjenta na żywo (wszystkie po kolei)",
            font=("Arial", 10, "bold"), bg="#c8e6c9", height=2, width=48,
            command=self.obsluz_przycisk_sesji
        )
        self.btn_sesja.pack(pady=5)

    def odswiez_liste_cwiczen(self):
        self.listbox_cwiczenia.delete(0, tk.END)
        dane = db.pobierz_liste_cwiczen()
        self.lista_cwiczen = [nazwa for nazwa, _ in dane]
        for nazwa, ma_ideal in dane:
            status = "✔ [Ideał gotowy]" if ma_ideal else "✖ [Brak ideału]"
            self.listbox_cwiczenia.insert(tk.END, f"{nazwa}  {status}")

    def dodaj_cwiczenie(self):
        nazwa = simpledialog.askstring("Nowe ćwiczenie", "Podaj nazwę ćwiczenia (np. 01_Przysiad):")
        if not nazwa:
            return
        utworzona = db.utworz_cwiczenie(nazwa)
        if utworzona:
            self.odswiez_liste_cwiczen()
            self.status_label.config(text=f"Dodano ćwiczenie: {utworzona}. Ustaw wagi i nagraj ideał.")

    def usun_cwiczenie(self):
        wybor = self.listbox_cwiczenia.curselection()
        if not wybor:
            return
        nazwa = self.lista_cwiczen[wybor[0]]
        if messagebox.askyesno("Potwierdzenie", f"Usunąć ćwiczenie '{nazwa}' i wszystkie jego nagrania?"):
            db.usun_cwiczenie(nazwa)
            self.odswiez_liste_cwiczen()

    def otworz_okno_wag(self):
        wybor = self.listbox_cwiczenia.curselection()
        if not wybor:
            messagebox.showwarning("Brak wyboru", "Zaznacz ćwiczenie na liście, aby ustawić jego wagi.")
            return
        nazwa = self.lista_cwiczen[wybor[0]]
        OknoWagStawow(self.root, nazwa)

    def generuj_pdf_dla_zaznaczonego(self):
        wybor = self.listbox_cwiczenia.curselection()
        if not wybor:
            messagebox.showwarning("Brak wyboru", "Zaznacz ćwiczenie na liście, aby wygenerować raport PDF.")
            return
        nazwa = self.lista_cwiczen[wybor[0]]
        sciezka_pdf = db.generuj_raport_pdf(nazwa)
        if not sciezka_pdf:
            messagebox.showinfo("Brak danych", f"Ćwiczenie '{nazwa}' nie ma jeszcze zapisanych prób pacjenta.")
            return

        self.status_label.config(text=f"Wygenerowano raport PDF: {os.path.basename(sciezka_pdf)}", fg="#00695c")
        if messagebox.askyesno("Raport gotowy", f"Zapisano raport:\n{sciezka_pdf}\n\nCzy otworzyć plik PDF teraz?"):
            os.startfile(sciezka_pdf)

    # --- ANALIZA PLIKÓW WIDEO ---

    def _przetworz_plik_wideo(self, sciezka_wideo, nazwa_cwiczenia):
        cap_plik = cv2.VideoCapture(sciezka_wideo)
        if not cap_plik.isOpened():
            messagebox.showerror("Błąd pliku", "Nie udało się otworzyć wybranego pliku wideo.")
            return None, None, DOMYSLNE_FPS

        fps_pliku = cap_plik.get(cv2.CAP_PROP_FPS) or DOMYSLNE_FPS
        liczba_klatek = int(cap_plik.get(cv2.CAP_PROP_FRAME_COUNT))
        sekwencja, klatki_wynikowe = [], []
        nr_klatki = 0

        self.silnik.resetuj_filtr()
        self.tryb_nagrywania = "analiza_pliku"

        while True:
            ret, frame = cap_plik.read()
            if not ret:
                break
            nr_klatki += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cechy = self.silnik.analizuj_klatke(rgb)
            if cechy is not None:
                sekwencja.append(cechy)
            klatki_wynikowe.append(rgb.copy())

            if nr_klatki % 3 == 0 or nr_klatki == liczba_klatek:
                info = f"{nr_klatki}/{liczba_klatek}" if liczba_klatek > 0 else f"{nr_klatki}"
                self.status_label.config(text=f"Analiza wideo dla '{nazwa_cwiczenia}': klatka {info}...", fg="#6a1b9a")
                podglad = cv2.resize(rgb, (640, 480)) if (rgb.shape[1] > 800 or rgb.shape[0] > 600) else rgb
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(podglad))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.root.update()

        cap_plik.release()
        self.silnik.resetuj_filtr()
        self.tryb_nagrywania = None
        return sekwencja, klatki_wynikowe, fps_pliku

    def wczytaj_wideo_jako_ideal(self):
        if self.tryb_nagrywania is not None:
            return
        wybor = self.listbox_cwiczenia.curselection()
        if not wybor:
            messagebox.showwarning("Brak wyboru", "Zaznacz ćwiczenie na liście.")
            return

        sciezka = filedialog.askopenfilename(filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov *.mkv")])
        if not sciezka:
            return

        nazwa = self.lista_cwiczen[wybor[0]]
        sekwencja, klatki, fps = self._przetworz_plik_wideo(sciezka, nazwa)
        if not sekwencja:
            messagebox.showwarning("Brak detekcji", "Nie wykryto sylwetki na nagraniu!")
            return

        db.zapisz_ideal(nazwa, sekwencja, klatki, fps)
        self.odswiez_liste_cwiczen()
        self.status_label.config(text=f"Zapisano ideał 3D z pliku dla '{nazwa}'.", fg="#1b5e20")

    def analizuj_wideo_pacjenta(self):
        if self.tryb_nagrywania is not None:
            return
        wybor = self.listbox_cwiczenia.curselection()
        if not wybor:
            messagebox.showwarning("Brak wyboru", "Zaznacz ćwiczenie na liście.")
            return

        nazwa = self.lista_cwiczen[wybor[0]]
        ideal_dane = db.wczytaj_ideal(nazwa)
        if not ideal_dane:
            messagebox.showwarning("Brak wzorca", f"Ćwiczenie '{nazwa}' nie ma jeszcze wzorca!")
            return

        sciezka = filedialog.askopenfilename(filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov *.mkv")])
        if not sciezka:
            return

        sekwencja, klatki, fps = self._przetworz_plik_wideo(sciezka, nazwa)
        if not sekwencja:
            messagebox.showwarning("Brak detekcji", "Nie wykryto sylwetki pacjenta!")
            return

        wagi = db.wczytaj_wagi_cwiczenia(nazwa)
        ocena, bledy = db.ocen_ruch_szczegolowo(ideal_dane, sekwencja, wagi)
        czas = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        db.zapisz_probke_pacjenta(nazwa, czas, sekwencja, klatki, ocena, bledy, fps, prefiks="probka_z_pliku")

        opis_bledow = "\n".join([f"  - {staw}: odchylenie ~{kat}°" for staw, kat in bledy]) or "  Brak istotnych odchyleń!"
        self.status_label.config(text=f"Wynik dla '{nazwa}': {ocena}%", fg="#1b5e20")
        messagebox.showinfo(
            "Wynik analizy nagrania",
            f"Ćwiczenie: {nazwa}\nZgodność z ideałem (ważona 3D): {ocena}%\n\nNajwiększe odchylenia:\n{opis_bledow}"
        )

    # --- TIMERY I KAMERA NA ŻYWO ---

    def _uruchom_odliczanie_startu(self, sekundy, nazwa_cwiczenia, callback):
        self.odliczanie = sekundy
        self.nazwa_na_ekranie = nazwa_cwiczenia
        if sekundy > 0:
            self.status_label.config(text=f"Ćwiczenie: {nazwa_cwiczenia} — start za {sekundy}s...", fg="#d84315")
            self.timer_start_id = self.root.after(1000, lambda: self._uruchom_odliczanie_startu(sekundy - 1, nazwa_cwiczenia, callback))
        else:
            self.timer_start_id = None
            self.silnik.resetuj_filtr()
            callback()

    def _odliczaj_czas_idealu(self, pozostalo):
        self.pozostaly_czas_idealu = pozostalo
        if self.tryb_nagrywania != "ideal":
            self.pozostaly_czas_idealu = 0
            return
        if pozostalo > 0:
            self.status_label.config(text=f"● NAGRYWANIE IDEAŁU ({self.aktualne_cwiczenie_idealu}): {pozostalo}s", fg="#b71c1c")
            self.timer_ideal_id = self.root.after(1000, lambda: self._odliczaj_czas_idealu(pozostalo - 1))
        else:
            self.timer_ideal_id = None
            self._zakoncz_i_zapisz_ideal()

    def _anuluj_timery(self):
        for tid in (self.timer_start_id, self.timer_ideal_id):
            if tid is not None:
                self.root.after_cancel(tid)
        self.timer_start_id = None
        self.timer_ideal_id = None
        self.odliczanie = 0
        self.pozostaly_czas_idealu = 0

    def _zakoncz_i_zapisz_ideal(self):
        self._anuluj_timery()
        self.tryb_nagrywania = None
        self.btn_ideal.config(text=f"Nagraj ideał z kamery ({CZAS_NAGRANIA_IDEALU}s)", bg="#d0e8ff")
        db.zapisz_ideal(self.aktualne_cwiczenie_idealu, self.aktualna_sekwencja, self.klatki_wideo)
        self.odswiez_liste_cwiczen()
        self.status_label.config(text=f"Zapisano wzorzec 3D dla: {self.aktualne_cwiczenie_idealu}", fg="#1b5e20")

    def przelacz_nagrywanie_idealu(self):
        if self.tryb_nagrywania in ("sesja", "analiza_pliku"):
            return
        if self.tryb_nagrywania == "ideal" and self.odliczanie > 0:
            self._anuluj_timery()
            self.tryb_nagrywania = None
            self.btn_ideal.config(text=f"Nagraj ideał z kamery ({CZAS_NAGRANIA_IDEALU}s)", bg="#d0e8ff")
            self.status_label.config(text="Anulowano nagrywanie.", fg="#1a237e")
            return
        if self.tryb_nagrywania == "ideal":
            self._zakoncz_i_zapisz_ideal()
            return

        wybor = self.listbox_cwiczenia.curselection()
        if not wybor:
            messagebox.showwarning("Brak wyboru", "Zaznacz ćwiczenie na liście.")
            return

        self.aktualne_cwiczenie_idealu = self.lista_cwiczen[wybor[0]]
        self.aktualna_sekwencja, self.klatki_wideo = [], []
        self.tryb_nagrywania = "ideal"
        self.btn_ideal.config(text="Zatrzymaj / Anuluj", bg="#ffcdd2")
        self._uruchom_odliczanie_startu(3, self.aktualne_cwiczenie_idealu, lambda: self._odliczaj_czas_idealu(CZAS_NAGRANIA_IDEALU))

    def obsluz_przycisk_sesji(self):
        if self.tryb_nagrywania in ("ideal", "analiza_pliku"):
            return
        if self.tryb_nagrywania == "sesja" and self.odliczanie > 0:
            self._anuluj_timery()
            self.tryb_nagrywania = None
            self.btn_sesja.config(text="▶ Rozpocznij próbę pacjenta na żywo (wszystkie po kolei)", bg="#c8e6c9")
            self.status_label.config(text="Anulowano sesję.", fg="#1a237e")
            return

        if self.tryb_nagrywania != "sesja":
            gotowe = [nazwa for nazwa, ma_ideal in db.pobierz_liste_cwiczen() if ma_ideal]
            if not gotowe:
                messagebox.showwarning("Brak wzorców", "Żadne ćwiczenie nie ma jeszcze wzorca!")
                return
            self.cwiczenia_do_sesji = gotowe
            self.indeks_sesji = 0
            self.wyniki_sesji = []
            self.czas_sesji = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._rozpocznij_kolejne_cwiczenie_w_sesji()
        else:
            self._zakoncz_biezace_cwiczenie_w_sesji()

    def _rozpocznij_kolejne_cwiczenie_w_sesji(self):
        nazwa = self.cwiczenia_do_sesji[self.indeks_sesji]
        self.aktualna_sekwencja, self.klatki_wideo = [], []
        self.licznik_klatek_koncowych = 0
        self.tryb_nagrywania = "sesja"

        self.biezacy_ideal_dane = db.wczytaj_ideal(nazwa)
        self.biezace_wagi = db.wczytaj_wagi_cwiczenia(nazwa)
        self.docelowa_pozycja_koncowa = np.mean(self.biezacy_ideal_dane[-5:], axis=0) if self.biezacy_ideal_dane else None

        nr, razem = self.indeks_sesji + 1, len(self.cwiczenia_do_sesji)
        self.btn_sesja.config(text="Anuluj podczas odliczania...", bg="#eeeeee")

        def start():
            self.status_label.config(text=f"● SESJA ({nr}/{razem}): Wykonaj '{nazwa}'", fg="#e65100")
            etykieta = f"Zakończ ręcznie '{nazwa}' -> następne ({nr+1}/{razem})" if nr < razem else f"Zakończ ręcznie '{nazwa}' -> podsumowanie"
            self.btn_sesja.config(text=etykieta, bg="#ffe0b2" if nr < razem else "#ffccbc")

        self._uruchom_odliczanie_startu(3, f"{nr}/{razem}: {nazwa}", start)

    def _zakoncz_biezace_cwiczenie_w_sesji(self):
        nazwa = self.cwiczenia_do_sesji[self.indeks_sesji]
        ocena, bledy = db.ocen_ruch_szczegolowo(self.biezacy_ideal_dane, self.aktualna_sekwencja, self.biezace_wagi)
        db.zapisz_probke_pacjenta(nazwa, self.czas_sesji, self.aktualna_sekwencja, self.klatki_wideo, ocena, bledy)

        self.wyniki_sesji.append((nazwa, ocena, bledy))
        self.indeks_sesji += 1

        if self.indeks_sesji < len(self.cwiczenia_do_sesji):
            self._rozpocznij_kolejne_cwiczenie_w_sesji()
        else:
            self.tryb_nagrywania = None
            self.btn_sesja.config(text="▶ Rozpocznij próbę pacjenta na żywo (wszystkie po kolei)", bg="#c8e6c9")
            self.status_label.config(text="Sesja pacjenta zakończona i zapisana!", fg="#1b5e20")

            linie = []
            for cw, wynik, bl in self.wyniki_sesji:
                uwagi = f" (uwaga na: {', '.join(s for s, _ in bl)})" if bl else ""
                linie.append(f"• {cw}: {wynik}%{uwagi}")
            srednia = sum(w for _, w, _ in self.wyniki_sesji) / len(self.wyniki_sesji)
            messagebox.showinfo("Podsumowanie sesji", f"{chr(10).join(linie)}\n\nŚrednia sesji: {srednia:.1f}%")

    def aktualizuj_klatke(self):
        if self.tryb_nagrywania != "analiza_pliku":
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape

                cechy = self.silnik.analizuj_klatke(rgb)
                faktyczne_nagrywanie = (self.tryb_nagrywania in ("ideal", "sesja")) and (self.odliczanie == 0)
                wykryto_koniec = False

                if cechy is not None and faktyczne_nagrywanie:
                    self.aktualna_sekwencja.append(cechy)

                    if (
                        self.tryb_nagrywania == "sesja"
                        and self.docelowa_pozycja_koncowa is not None
                        and len(self.aktualna_sekwencja) > MIN_KLATEK_PRZED_STOPEM
                    ):
                        # Ważone porównanie pozycji końcowej (ignoruje mało istotne stawy)
                        wagi_arr = np.array(self.biezace_wagi, dtype=float)
                        if np.sum(wagi_arr) > 0:
                            roznica = np.average(np.abs(np.array(cechy) - self.docelowa_pozycja_koncowa), weights=wagi_arr)
                        else:
                            roznica = np.mean(np.abs(np.array(cechy) - self.docelowa_pozycja_koncowa))

                        self.licznik_klatek_koncowych = self.licznik_klatek_koncowych + 1 if roznica <= PROG_ZGODNOSCI_KONCA else 0
                        if self.licznik_klatek_koncowych >= WYMAGANE_KLATKI_KONCA:
                            wykryto_koniec = True

                if faktyczne_nagrywanie:
                    self.klatki_wideo.append(rgb.copy())
                    cv2.circle(rgb, (30, 30), 12, (255, 0, 0), -1)
                    if self.tryb_nagrywania == "ideal" and self.pozostaly_czas_idealu > 0:
                        rysuj_tekst_z_obrysem(rgb, f"REC: {self.pozostaly_czas_idealu}s", (55, 40), 0.9, (255, 80, 80), 2)

                if self.odliczanie > 0:
                    rysuj_tekst_z_obrysem(rgb, f"Cwiczenie: {self.nazwa_na_ekranie}", (30, 60), 1.0, (0, 255, 255), 2)
                    rysuj_tekst_z_obrysem(rgb, str(self.odliczanie), (w // 2 - 45, h // 2 + 55), 5.0, (255, 215, 0), 10)

                imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                if wykryto_koniec:
                    self._zakoncz_biezace_cwiczenie_w_sesji()

        self.root.after(20, self.aktualizuj_klatke)

    def zamknij(self):
        self._anuluj_timery()
        self.cap.release()
        self.root.destroy()