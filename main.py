import cv2
from ultralytics import YOLO

# Загружаем модель
model = YOLO("yolov8n.pt")  # n – самая лёгкая, можно s/m/l

# Открываем видео
cap = cv2.VideoCapture(r"D:\DOCS\DIPLOM\videos\AVR_4D24B_.mp4")  # или 0 для веб-камеры

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Запуск детекции
    results = model(frame, classes=[0], conf=0.4) # classes=[0] - детектить  только людей

    # Отрисовка результатов
    annotated_frame = results[0].plot()

    # Показ видео
    cv2.imshow("YOLO – person detection", annotated_frame)

    # Выход по клавише Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()