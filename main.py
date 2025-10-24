import os #operacje na plikach, folderach, ścieżkach itp
from absl import logging
import warnings
import math
import cv2
import random
from collections import deque
import mediapipe as mp  # [MP] biblioteka do analizy dłoni
print("MediaPipe działa poprawnie!")

# ============================================================
#  KONFIGURACJA
# ============================================================

# Wyłączenie logów TensorFlow i MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
logging.set_verbosity(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Feedback manager*")

# [MP] Inicjalizacja MediaPipe Hands
mp_hands = mp.solutions.hands #pobiera moduł hands z pakietu mediapipe do ykrywania i śledzenia dłoni
mp_drawing = mp.solutions.drawing_utils #pobiera moduł drawing_utils z mediapipe do rysowania landmarków i połączeń dłoni na obrazie (linie i punkty)

# ============================================================
#  FUNKCJE POMOCNICZE
# ============================================================

def ekstraktuj_punkty(hand_landmarks):
    """Zwraca znormalizowane punkty dłoni (x, y) względem nadgarstka."""
    punkty = [(lm.x, lm.y) for lm in hand_landmarks.landmark] #Pobiera współrzędne wszystkich punktów dłoni
    ref_x, ref_y = punkty[0]
    punkty_shifted = [(x - ref_x, y - ref_y) for x, y in punkty] #Przesuwa układ odniesienia tak, żeby punkt 0 (nadgarstek) był w (0, 0)

    base_len = math.dist(punkty[0], punkty[9]) or 1.0 #Normalizuje rozmiar dłoni — dzieli przez odległość między punktem 0 (nadgarstek) a 9 (środek dłoni).
    return [(x / base_len, y / base_len) for x, y in punkty_shifted] #Zwraca listę punktów (x, y) znormalizowanych — czyli niezależnych od odległości i położenia.


def porownaj_szkielety(szk1, szk2):
    """Porównuje dwa szkielety dłoni — im mniejszy dystans, tym większe podobieństwo."""
    if len(szk1) != len(szk2):
        return float("inf")
    return sum(math.dist(a, b) for a, b in zip(szk1, szk2)) / len(szk1)
#Liczy średnią odległość między odpowiadającymi sobie punktami, im mniejsza wartość tym bardziej podobne gesty

# ============================================================
#  WCZYTYWANIE WZORCÓW STATYCZNYCH
# ============================================================

def wczytaj_wzorce(folder_path):
    """Wczytuje obrazy wzorców gestów statycznych z folderu."""
    print(f"📂 Wczytywanie wzorców z: {folder_path}")
    wzorce = {}

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands: #Jeśli znajdzie dłoń
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            frame = cv2.imread(path)
            if frame is None:
                print(f"⚠ Nie można wczytać pliku: {filename}")
                continue

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                print(f"⚠ Brak dłoni w {filename}")
                continue

            label = os.path.splitext(filename)[0].lower() #etykieta gestu to nazwa pliku bez rozszerzenia
            wzorce[label] = ekstraktuj_punkty(results.multi_hand_landmarks[0])  #zapisuje jej znormalizowany szkielet
            print(f"✅ Załadowano: {label}")

    print(f"📁 Łącznie {len(wzorce)} wzorców wczytanych.\n")
    return wzorce

# ============================================================
#  WCZYTYWANIE WZORCÓW DYNAMICZNYCH
# ============================================================

def wczytaj_wzorce_dynamiczne(folder_path):
    """
    Wczytuje wzorce gestów dynamicznych z plików wideo.
    Każdy wzorzec jest listą klatek, a każda klatka to lista punktów dłoni.
    """
    print(f"🎥 Wczytywanie wzorców dynamicznych z: {folder_path}")
    wzorce_dyn = {}

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"⚠ Nie można otworzyć pliku: {filename}")
                continue

            sekwencja = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    punkty = ekstraktuj_punkty(hand_landmarks)
                    sekwencja.append(punkty)

            cap.release()
            label = os.path.splitext(filename)[0].lower()
            if sekwencja:
                wzorce_dyn[label] = sekwencja
                print(f"✅ Załadowano gest dynamiczny: {label} ({len(sekwencja)} klatek)")
            else:
                print(f"⚠ Brak dłoni w wideo: {filename}")

    print(f"📁 Łącznie {len(wzorce_dyn)} wzorców dynamicznych.\n")
    return wzorce_dyn


# Funkcja "zaślepka" – klasyfikacja statyczna
#with to konstrukcja w Pythonie używana do zarządzania kontekstem.
# Oznacza to, że automatycznie wykonuje pewne czynności przy wejściu i wyjściu z bloku kodu.
def klasyfikuj_stat(frame, wzorce_stat):
    max_allowed_distance = 0.2 # spróbowac dobrać
    # [MP] Utwórz obiekt Hands dla pojedynczej klatki (statyczny gest)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        # [MP] Konwersja obrazu na RGB dla mediapipe zamiast BGR
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # [MP] Jeśli wykryto dłoń – narysuj szkielet
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
				# --- Klasyfikacja statyczna ---
                szkiel_test = ekstraktuj_punkty(hand_landmarks) #Oblicza aktualny szkielet
                najlepszy, min_dist = "-", float("inf")

                for litera, szkiel_wzorzec in wzorce_stat.items():
                    dist = porownaj_szkielety(szkiel_test, szkiel_wzorzec) #Porównuje go z każdym wzorcem
                    if dist < min_dist:
                        najlepszy, min_dist = litera, dist
                if min_dist > max_allowed_distance:
                    return '-'
                return najlepszy

    return '-'

# Funkcja "zaślepka" – klasyfikacja dynamiczna
def klasyfikuj_dyn(buffer, wzorce_dyn):
    """
    Klasyfikuje ruch dłoni jako gest dynamiczny,
    porównując przebieg klatek z wzorcami dynamicznymi.
    """
    # Utworzenie szkieletów z bufora bieżącego
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
        test_seq = []
        for frame in buffer:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                test_seq.append(ekstraktuj_punkty(hand_landmarks))

    if not test_seq:
        return '-'

    # Proste dopasowanie sekwencji z karą za różnicę długości
    def porownaj_sekwencje(seq1, seq2):
        min_len = min(len(seq1), len(seq2))
        dystanse = [porownaj_szkielety(seq1[i], seq2[i]) for i in range(min_len)]
        sredni = sum(dystanse) / len(dystanse)
        kara_dlugosc = abs(len(seq1) - len(seq2)) * 0.02
        return sredni + kara_dlugosc

    najlepszy, min_dist = '-', float('inf')
    for gest, seq_wzorzec in wzorce_dyn.items():
        dist = porownaj_sekwencje(test_seq, seq_wzorzec)
        if dist < min_dist:
            min_dist = dist
            najlepszy = gest

    # Próg dopasowania (dobierany eksperymentalnie)
    if min_dist > 0.3:
        return '-'
    return najlepszy


def przetworz_video(video_path, wzorce_stat, wzorce_dyn):
    print(f"\n▶️ Rozpoczynam analizę pliku: {video_path}")
    bufor = deque(maxlen=25)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Nie można otworzyć pliku wideo: {video_path}")
        return

    # gesty, które mogą być początkiem dynamicznych
    potencjalnie_dynamiczne = {'a','c','e','i','l','n','o','r','s'}

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"🎬 Koniec filmu: {video_path}")
            break

        bufor.append(frame)
        if len(bufor) < bufor.maxlen:
            cv2.imshow("Podgląd", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        wynik_stat = klasyfikuj_stat(bufor[0], wzorce_stat)
        if wynik_stat == '-':
            cv2.imshow("Podgląd", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        if wynik_stat in potencjalnie_dynamiczne:
            wynik_dyn = klasyfikuj_dyn(list(bufor), wzorce_dyn)
            if wynik_dyn != '-':
                print(f"💫 Wykryto gest dynamiczny: {wynik_dyn}")
                bufor.clear()
            else:
                print(f"👌 Wykryto gest statyczny: {wynik_stat}")
        else:
            print(f"👌 Wykryto gest statyczny: {wynik_stat}")

        cv2.imshow("Podgląd", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
#  MAIN
# ============================================================

def main():
    videos = ["pfa2.mp4"]

    wzorce_stat = wczytaj_wzorce(r"C:\\Users\\Ola Sawicka\\Desktop\\semestr 7\\thesis\\statycze")
    wzorce_dyn = wczytaj_wzorce_dynamiczne(r"C:\\Users\\Ola Sawicka\\Desktop\\semestr 7\\thesis\\dynamiczne")

    for video in videos:
        przetworz_video(video, wzorce_stat, wzorce_dyn)

    print("\n✅ Wszystkie pliki zostały przetworzone!")


if __name__ == "__main__": #uruchomił main() tylko wtedy, gdy ten plik został uruchomiony bezpośrednio, a nie np. zaimportowany z innego pliku.
    main()