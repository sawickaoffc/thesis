import os #operacje na plikach, folderach, ścieżkach itp
import math
import random
import cv2 #biblioteka OpenCV do obsługi obrazu i wideo
from collections import deque #dwustronna kolejka (bufor klatek)
import mediapipe as mp
print("MediaPipe działa poprawnie!")

# ============================================================
#  KONFIGURACJA
# ============================================================

# Wyłączenie logów TensorFlow i MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands #model do wykrywania dłoni
mp_drawing = mp.solutions.drawing_utils #narzędzia do rysowania szkieletu dłoni na obrazie


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
                print(f"⚠️ Nie można wczytać pliku: {filename}")
                continue

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                print(f"⚠️ Brak dłoni w {filename}")
                continue

            label = os.path.splitext(filename)[0].lower() #etykieta gestu to nazwa pliku bez rozszerzenia
            wzorce[label] = ekstraktuj_punkty(results.multi_hand_landmarks[0])  #zapisuje jej znormalizowany szkielet
            print(f"✅ Załadowano: {label}")

    print(f"📁 Łącznie {len(wzorce)} wzorców wczytanych.\n")
    return wzorce


# ============================================================
#  PRZETWARZANIE WIDEO
# ============================================================

def przetworz_video(video_path, wzorce_stat):
    """Analizuje pojedynczy plik wideo i rozpoznaje gesty dłoni."""
    print(f"\n▶️ Analiza pliku: {video_path}")

    if not os.path.exists(video_path):
        print(f"❌ Nie znaleziono pliku: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć: {video_path}")
        return

    # Bufor klatek i stabilizacja
    bufor = deque(maxlen=25)
    stable_stat = {"last": None, "count": 0}
    stable_dyn = {"last": None, "count": 0}

    # Jeden obiekt Hands dla całego wideo
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.9
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"🎬 Koniec filmu: {video_path}")
                break

            bufor.append(frame)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #Rysuje punkty i połączenia na obrazie

                # --- Klasyfikacja statyczna ---
                szkiel_test = ekstraktuj_punkty(hand_landmarks) #Oblicza aktualny szkielet
                najlepszy, min_dist = "-", float("inf")

                for litera, szkiel_wzorzec in wzorce_stat.items():
                    dist = porownaj_szkielety(szkiel_test, szkiel_wzorzec) #Porównuje go z każdym wzorcem
                    if dist < min_dist:
                        najlepszy, min_dist = litera, dist

                # --- Stabilizacja wyniku ---
                if najlepszy == stable_stat["last"]:
                    stable_stat["count"] += 1
                else:
                    stable_stat["count"] = 0
                stable_stat["last"] = najlepszy

                #wymaga żeby ten sam gest pojawił się kilka razy z rzędu
                if stable_stat["count"] >= 15:
                    print(f"✋ Statyczny gest: {najlepszy}")
                    stable_stat["count"] = 0

                # --- Klasyfikacja dynamiczna (zaślepka) ---
                if len(bufor) > 1:
                    dyn_gest = random.choice(["Ą", "Ę", "Ł", "Ó", "-", "Ś", "Ź", "Ż"])
                    if dyn_gest == stable_dyn["last"]:
                        stable_dyn["count"] += 1
                    else:
                        stable_dyn["count"] = 0
                    stable_dyn["last"] = dyn_gest

                    if stable_dyn["count"] >= 5:
                        print(f"🎯 Dynamiczny gest: {dyn_gest}")
                        stable_dyn["count"] = 0

            # Podgląd wideo
            cv2.imshow("Podgląd", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#  MAIN
# ============================================================

def main():
    videos = ["pfa2.mp4"]
    wzorce_stat = wczytaj_wzorce(r"C:\Users\Ola Sawicka\Desktop\semestr 7\thesis\statycze")

    for video in videos:
        przetworz_video(video, wzorce_stat)

    print("\n✅ Wszystkie pliki zostały przetworzone!")


if __name__ == "__main__":
    main()
