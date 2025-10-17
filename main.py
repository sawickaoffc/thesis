import cv2
import random
from collections import deque
import mediapipe as mp  # [MP] biblioteka do analizy dłoni

# [MP] Inicjalizacja MediaPipe Hands
mp_hands = mp.solutions.hands #pobiera moduł hands z pakietu mediapipe do ykrywania i śledzenia dłoni
mp_drawing = mp.solutions.drawing_utils #pobiera moduł drawing_utils z mediapipe do rysowania landmarków i połączeń dłoni na obrazie (linie i punkty)


# Funkcja "zaślepka" – klasyfikacja statyczna
#with to konstrukcja w Pythonie używana do zarządzania kontekstem.
# Oznacza to, że automatycznie wykonuje pewne czynności przy wejściu i wyjściu z bloku kodu.
def klasyfikuj_stat(frame):
    # [MP] Utwórz obiekt Hands dla pojedynczej klatki (statyczny gest)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        # [MP] Konwersja obrazu na RGB dla mediapipe zamiast BGR
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # [MP] Jeśli wykryto dłoń – narysuj szkielet
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Losowy wybór litery statycznej albo brak ('-')
    litery_stat = ['A', 'B', 'C', 'D', 'E', '-', 'F', 'G', 'H']
    return random.choice(litery_stat)

# Funkcja "zaślepka" – klasyfikacja dynamiczna
def klasyfikuj_dyn(buffer):
    # Utwórz obiekt Hands do przetwarzania sekwencji klatek (ruch dłoni)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3) as hands:
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

def przetworz_video(video_path):
    #Przetwarza pojedynczy plik wideo
    print(f"\n▶️ Rozpoczynam analizę pliku: {video_path}") #dodałam emoji!
    bufor = deque(maxlen=25) #bufor stworzony
    cap = cv2.VideoCapture(video_path) #odczytujemy ttym razem z pliku a nie z kamery

    if not cap.isOpened():
        print(f"❌ Nie można otworzyć pliku wideo: {video_path}")
        return

    while True:
        ret, frame = cap.read() #odczytywanie tylko jednej klatki!
        if not ret:
            print(f"🎬 Koniec filmu: {video_path}")
            break

        # Dodaj klatkę do bufora
        bufor.append(frame)

        # Jeśli bufor jest jeszcze niepełny, kontynuuj
        if len(bufor) < bufor.maxlen:
            cv2.imshow("Podgląd", frame) #dzięki temu widzimy nasz filmik obok wyświetlony
            if cv2.waitKey(30) & 0xFF == ord('q'): #dałam trovchę więcej czasu na przeczekanie, było 1ms ale teraz myślę że lepiej jak wstrzyma troszke dłużej
                break
            continue

        # Klasyfikacja statyczna
        wynik_stat = klasyfikuj_stat(bufor[0])

        if wynik_stat == '-':
            cv2.imshow("Podgląd", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        # Sprawdź, czy znak statyczny może być początkiem dynamicznego
        potencjalnie_dynamiczne = {'A', 'B', 'C'} # przykładowy zbiór
        if wynik_stat in potencjalnie_dynamiczne:
            wynik_dyn = klasyfikuj_dyn(list(bufor))
            if wynik_dyn != '-':
                print(f"💫 Wykryto gest dynamiczny: {wynik_dyn}")
                bufor.clear() # wyczyść bufor po dynamicznym
            else:
                print(f"👌 Wykryto gest statyczny: {wynik_stat}")
        else:
            print(f"👌 Wykryto gest statyczny: {wynik_stat}")

        # Wyświetl podgląd z kamery
        cv2.imshow("Podgląd", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # 🔹 Lista plików wideo do przetworzenia
    videos = [
        "pfa1.avi",
        "pfa2.mp4",
        "pfa3.avi",
        "pfa4.avi"
    ]

    for video in videos:
        przetworz_video(video)

    print("\n✅ Wszystkie pliki zostały przetworzone!")

if __name__ == "__main__": #uruchomił main() tylko wtedy, gdy ten plik został uruchomiony bezpośrednio, a nie np. zaimportowany z innego pliku.
    main()