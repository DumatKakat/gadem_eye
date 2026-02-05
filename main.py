import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Создание детектора
model_path = '.\\hand_landmarker.task'  # Нужно скачать модель
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,  # Режим для видео
    num_hands=2  # Максимум 2 руки
)
detector = HandLandmarker.create_from_options(options)

# 2. Обработка видео с камеры
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертация кадра в формат MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Обнаружение рук
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # 3. Отрисовка результатов (логика похожа на старую)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Здесь нужно преобразовать landmarks и нарисовать их с помощью cv2.circle и cv2.line
            # Пример для отрисовки точек:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()