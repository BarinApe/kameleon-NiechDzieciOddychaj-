KineTrack 3D — System Analizy Biomechanicznej i Detekcji Kompensacji Ruchu

1. Instalacja bibliotek:
   pip install mediapipe opencv-python numpy pillow matplotlib

2. Uruchomienie programu:
   python app_kinetrack.py

3. Zawartość paczki:
   - app_kinetrack.py       (główny program z interfejsem i wizualizacją kątów na szkielecie)
   - kinematics_3d.py       (silnik analizy 3D z filtrem One-Euro i detekcją 14 kompensacji)
   - patient_tracker.py     (moduł oceny MQI i zapisu historii postępów pacjenta)
   - biomechanics_config.py (konfiguracja kątów stawowych i progów kompensacji)
