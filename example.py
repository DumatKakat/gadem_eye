import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Функция для подсчета пальцев по 21 ключевой точке
def count_fingers_from_landmarks(landmarks, handedness="Right"):
    """
    Функция подсчета поднятых пальцев для новой версии MediaPipe.
    landmarks: список объектов NormalizedLandmark (с атрибутами x, y, z)
    handedness: строка ("Right" или "Left") для учета типа руки
    """
    tip_ids = [4, 8, 12, 16, 20]  # Кончики пальцев: большой, указательный, средний, безымянный, мизинец
    pip_ids = [2, 6, 10, 14, 18]  # Средние суставы (для сравнения)

    fingers = []

    # Большой палец (сравниваем по X-координате)
    thumb_tip = landmarks[tip_ids[0]]
    thumb_pip = landmarks[pip_ids[0]]

    # Учитываем, левая это рука или правая
    if handedness == "Right":
        # Для правой руки: если кончик левее сустава, палец поднят
        fingers.append(1 if thumb_tip.x < thumb_pip.x else 0)
    else:
        # Для левой руки: если кончик правее сустава, палец поднят
        fingers.append(1 if thumb_tip.x > thumb_pip.x else 0)

    # Остальные 4 пальца (сравниваем по Y-координате)
    for i in range(1, 5):
        finger_tip = landmarks[tip_ids[i]]
        finger_pip = landmarks[pip_ids[i]]
        # Если кончик пальца выше сустава, палец считается поднятым
        fingers.append(1 if finger_tip.y < finger_pip.y else 0)

    return sum(fingers)

def main():
    # 1. Укажите ПРАВИЛЬНЫЙ путь к скачанному файлу модели
    model_path = '.\\hand_landmarker.task'  # ЗАМЕНИТЕ НА СВОЙ ПУТЬ

    # 2. Создание опций для HandLandmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,  # Режим для видео
        num_hands=3,  # Максимальное количество отслеживаемых рук
        min_hand_detection_confidence=0.8,  # Минимальная уверенность при детекции
        min_tracking_confidence=0.7  # Минимальная уверенность при отслеживании
    )

    # 3. Инициализация детектора
    detector = HandLandmarker.create_from_options(options)

    # 4. Захват видео с камеры
    cap = cv2.VideoCapture(0)

    # Переменная для хранения временной метки кадра
    frame_timestamp_ms = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Не удалось получить кадр с камеры.")
            break

        # 5. Конвертация кадра в формат MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # 6. Обнаружение рук в кадре
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 30  # Увеличиваем временную метку (примерно 30 мс на кадр)

        # 7. Обработка и отрисовка результатов
        if detection_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Определение типа руки (если доступно)
                handedness = "Right"  # значение по умолчанию
                if detection_result.handedness:
                    handedness = detection_result.handedness[hand_idx][0].display_name

                # Подсчет поднятых пальцев для этой руки
                fingers_count = count_fingers_from_landmarks(hand_landmarks, handedness)

                # Визуализация ключевых точек на кадре (простой вариант - точки)
                for landmark in hand_landmarks:
                    # Преобразование нормализованных координат в пиксельные
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # (Опционально) Можно нарисовать соединения между точками
                # Для этого потребуется знать порядок соединений (HAND_CONNECTIONS)

                # Вывод текста с количеством пальцев
                # Используем координату запястья (точка 0) как базовую для текста
                if hand_landmarks:
                    wrist = hand_landmarks[0]
                    text_x = int(wrist.x * frame.shape[1]) - 50
                    text_y = int(wrist.y * frame.shape[0]) - 30
                    cv2.putText(frame, f'Fingers: {fingers_count}',
                               (10,30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            path1 = '.\\img\\Udin_din_din_din_dun.jpg'
            path2 = '.\\img\\Ballerina_Cappuccina.png'

            if fingers_count == 1:
                img = cv2.imread(path1)
                cv2.imshow("Img", img)
                    # cv2.waitKey(0)
            if fingers_count == 2:
                img = cv2.imread(path2)
                cv2.imshow("Img", img)

        # 8. Отображение результата
        cv2.imshow('Hand Tracking - New MediaPipe API', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
# Первый коммит
# Первый коммит
# Первый коммит
# Первый коммит
# Первый коммит
# Первый коммит
# Первый коммит
