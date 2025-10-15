import cv2
import random
from collections import deque

# Funkcja "zaślepka" – klasyfikacja statyczna
def klasyfikuj_stat(frame):
    # Losowy wybór litery statycznej albo brak ('-')
    litery_stat = ['A', 'B', 'C', 'D', 'E', '-', 'F', 'G', 'H']
    return random.choice(litery_stat)

# Funkcja "zaślepka" – klasyfikacja dynamiczna
def klasyfikuj_dyn(buffer):
    # Losowy wybór litery dynamicznej albo brak ('-')
    litery_dyn = ['Ą', 'Ę', 'Ł', 'Ó', '-', 'Ś', 'Ź', 'Ż']
    return random.choice(litery_dyn)

def  main():
    # Bufor na ostatnie 25 klatek
    bufor = deque(maxlen=25)

    cap = cv2.VideoCapture(0)  # 0 = kamera domyślna

    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return

    while True:
        ret, frame = cap.read() # odczytuje jedną klatkę
        if not ret:
            print("Błąd odczytu klatki z kamery")
            break

        # Dodaj klatkę do bufora
        bufor.append(frame)

        # Jeśli bufor jest jeszcze niepełny to wróć po następną klatkę
        if len(bufor) < bufor.maxlen:
            cv2.imshow("Podgląd", frame) #wyświetlam sobie obraz
            if cv2.waitKey(1) & 0xFF == ord('q'):  # warunek zatrzymania programu, klawisz Q
                break
            continue

        # Klasyfikacja statyczna na najstarszej klatce
        wynik_stat = klasyfikuj_stat(bufor[0])

        if wynik_stat == '-':
            # Brak gestu statycznego – wróć do zbierania klatek
            cv2.imshow("Podgląd", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Sprawdź, czy znak statyczny może być początkiem dynamicznego
        potencjalnie_dynamiczne = {'A', 'B', 'C'}  # przykładowy zbiór
        if wynik_stat in potencjalnie_dynamiczne:
            wynik_dyn = klasyfikuj_dyn(list(bufor))
            if wynik_dyn != '-':
                print("Wykryto gest dynamiczny:", wynik_dyn)
                bufor.clear()  # wyczyść bufor po dynamicznym
            else:
                print("Wykryto gest statyczny:", wynik_stat)
        else:
            print("Wykryto gest statyczny:", wynik_stat)

        # Wyświetl podgląd z kamery
        cv2.imshow("Podgląd", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": #uruchomił main() tylko wtedy, gdy ten plik został uruchomiony bezpośrednio, a nie np. zaimportowany z innego pliku.
    main()
