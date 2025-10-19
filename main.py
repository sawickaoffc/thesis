import os, sys
import contextlib

import mediapipe as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # wyłącza INFO i WARNING TensorFlow
os.environ['GLOG_minloglevel'] = '3'      # wyłącza INFO i WARNING MediaPipe C++
import math  # [DODANE] do obliczania odległości między punktami
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
import cv2
import random
from collections import deque
# [MP] Inicjalizacja MediaPipe Hands
mp_hands = mp.solutions.hands #pobiera moduł hands z pakietu mediapipe do ykrywania i śledzenia dłoni
mp_drawing = mp.solutions.drawing_utils #pobiera moduł drawing_utils z mediapipe do rysowania landmarków i połączeń dłoni na obrazie (linie i punkty)


# [DODANE] Funkcja do ekstrakcji i normalizacji punktów dłoni
def ekstraktuj_punkty(hand_landmarks):
    """Zwraca znormalizowane punkty dłoni jako listę (x, y), przesunięte i przeskalowane."""
    punkty = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    ref_x, ref_y = punkty[0]  # punkt odniesienia — nadgarstek
    punkty_shifted = [(x - ref_x, y - ref_y) for (x, y) in punkty]

    # Długość odniesienia — dystans między nadgarstkiem (0) a środkowym palcem (9)
    base_length = math.sqrt(
        (punkty[9][0] - punkty[0][0])**2 + (punkty[9][1] - punkty[0][1])**2
    )
    if base_length == 0:
        base_length = 1.0

    # Skalowanie
    punkty_norm = [(x / base_length, y / base_length) for (x, y) in punkty_shifted]
    return punkty_norm



# [DODANE] Funkcje pomocnicze do porównywania szkieletów
def dystans_euklidesowy(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for (a, b) in zip(p1, p2)))

def porownaj_szkielety(szkielet1, szkielet2):
    """Porównuje dwa szkielety dłoni — im mniejsza wartość, tym bardziej podobne."""
    if len(szkielet1) != len(szkielet2):
        return float('inf')
    dystans = sum(dystans_euklidesowy(a, b) for a, b in zip(szkielet1, szkielet2))
    return dystans / len(szkielet1)


# [DODANE] Funkcja wczytująca wzorce z folderu
def wczytaj_wzorce(folder_path):
    print(f"📂 Wczytywanie wzorców z folderu: {folder_path}")
    wzorce = {}
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            frame = cv2.imread(path)
            if frame is None:
                print(f"⚠️ Nie można wczytać pliku: {filename}")
                continue
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if not results.multi_hand_landmarks:
                print(f"⚠️ Brak dłoni w {filename}")
                continue
            hand_landmarks = results.multi_hand_landmarks[0]
            punkty = ekstraktuj_punkty(hand_landmarks)
            label = os.path.splitext(filename)[0].lower()  # np. 'a.jpg' -> 'a'
            wzorce[label] = punkty
            print(f"✅ Wczytano wzorzec: {label}")
    print(f"📁 Załadowano {len(wzorce)} wzorców statycznych\n")
    return wzorce


# Funkcja "zaślepka" – klasyfikacja statyczna
#with to konstrukcja w Pythonie używana do zarządzania kontekstem.
# Oznacza to, że automatycznie wykonuje pewne czynności przy wejściu i wyjściu z bloku kodu.
# Funkcja klasyfikująca statyczny gest z już utworzonym obiektem hands
def klasyfikuj_stat(frame, wzorce, hands):
    """
    frame: pojedyncza klatka obrazu (BGR)
    wzorce: słownik wzorców statycznych
    hands: obiekt mp_hands.Hands utworzony raz wcześniej
    """
    # Konwersja obrazu na RGB dla MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Ekstrakcja punktów dłoni i porównanie z wzorcami
        hand_landmarks = results.multi_hand_landmarks[0]
        szkiel_test = ekstraktuj_punkty(hand_landmarks)

        najlepszy_znak = '-'
        min_dystans = float('inf')
        for litera, szkiel_wzorzec in wzorce.items():
            dist = porownaj_szkielety(szkiel_test, szkiel_wzorzec)
            if dist < min_dystans:
                min_dystans = dist
                najlepszy_znak = litera

        return najlepszy_znak

    return '-'

# Funkcja "zaślepka" – klasyfikacja dynamiczna
def klasyfikuj_dyn(buffer):
    # Utwórz obiekt Hands do przetwarzania sekwencji klatek (ruch dłoni)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6) as hands:
        for frame in buffer:
            # Konwersja każdej klatki na RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Jeśli wykryto dłoń – narysuj szkielet
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Losowy wybór litery dynamicznej albo brak ('-')
    litery_dyn = ['Ą', 'Ę', 'Ł', 'Ó', '-', 'Ś', 'Ź', 'Ż']
    return random.choice(litery_dyn)


def przetworz_video(video_path, wzorce_stat):
    """Przetwarza wideo, wykrywając statyczne i dynamiczne gesty, szybko i stabilnie."""
    print(f"\n▶️ Analiza pliku: {video_path}")
    bufor = deque(maxlen=25)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć pliku wideo: {video_path}")
        return

    main.last_stat = None
    main.stable_stat_count = 0
    main.last_dyn = None
    main.stable_dyn_count = 0

    # Jeden obiekt Hands dla całego wideo
    with mp_hands.Hands(
        static_image_mode=False,  # False = wideo
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"🎬 Koniec filmu: {video_path}")
                break

            bufor.append(frame)

            # Konwersja na RGB i przetwarzanie klatki
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Rysowanie szkieletu dłoni
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- Klasyfikacja statyczna --- #
                szkiel_test = ekstraktuj_punkty(results.multi_hand_landmarks[0])
                najlepszy_stat = '-'
                min_dist = float('inf')
                for litera, szkiel_wzorzec in wzorce_stat.items():
                    dist = porownaj_szkielety(szkiel_test, szkiel_wzorzec)
                    if dist < min_dist:
                        min_dist = dist
                        najlepszy_stat = litera

                # Stabilizacja statyczna
                if najlepszy_stat == main.last_stat:
                    main.stable_stat_count += 1
                else:
                    main.stable_stat_count = 0
                main.last_stat = najlepszy_stat

                if main.stable_stat_count >= 16:
                    print(f"✋ Wykryto statyczny gest: {najlepszy_stat}")
                    main.stable_stat_count = 0

                # --- Klasyfikacja dynamiczna (ruch dłoni) --- #
                # Prosta heurystyka: porównanie odległości punktów z poprzednią klatką
                if len(bufor) > 1:
                    prev_frame = bufor[-2]
                    # Tu możesz dodać bardziej zaawansowaną analizę ruchu
                    # Na razie: losowy wybór dynamicznej litery dla testu
                    dyn_gest = random.choice(['Ą','Ę','Ł','Ó','-','Ś','Ź','Ż'])
                    if dyn_gest == main.last_dyn:
                        main.stable_dyn_count += 1
                    else:
                        main.stable_dyn_count = 0
                    main.last_dyn = dyn_gest

                    if main.stable_dyn_count >= 12:
                        print(f"🎯 Wykryto dynamiczny gest: {dyn_gest}")
                        main.stable_dyn_count = 0

            cv2.imshow("Podgląd", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # 🔹 Lista plików wideo do przetworzenia
    videos = [
        #"pfa1.avi",
        "pfa2.mp4",
        #"pfa3.avi",
        #"pfa4.avi"
    ]

    # [DODANE] Wczytanie wzorców przed rozpoczęciem analizy
    wzorce_stat = wczytaj_wzorce(r"C:\Users\Ola Sawicka\Desktop\semestr 7\thesis\statycze")

    for video in videos:
        przetworz_video(video, wzorce_stat)

    print("\n✅ Wszystkie pliki zostały przetworzone!")


if __name__ == "__main__": #uruchomił main() tylko wtedy, gdy ten plik został uruchomiony bezpośrednio, a nie np. zaimportowany z innego pliku.
    main()