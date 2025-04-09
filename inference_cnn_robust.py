import cv2
import numpy as np
import pyttsx3
import threading
from tensorflow.keras.models import load_model
from collections import deque
import mediapipe as mp

# Load the retrained CNN model
model = load_model('cnn_sign_model.h5')

# Label map (0 → A, ..., 24 → Y)
labels_dict = {i: chr(65 + i) for i in range(25)}

# Init TTS engine
engine = pyttsx3.init()
def speak(text):
    threading.Thread(target=lambda: [engine.say(text), engine.runAndWait()]).start()

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
drawing = mp.solutions.drawing_utils

# Video + buffers
cap = cv2.VideoCapture(0)
IMG_HEIGHT, IMG_WIDTH = 128, 128
predictions_buffer = deque(maxlen=7)
result_text = ""
last_spoken = ""
CONFIDENCE_THRESHOLD = 0.80

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Bounding box from landmarks
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = max(int(min(x_coords) * W) - 20, 0)
            y_min = max(int(min(y_coords) * H) - 20, 0)
            x_max = min(int(max(x_coords) * W) + 20, W)
            y_max = min(int(max(y_coords) * H) + 20, H)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            roi_resized = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
            roi_input = np.expand_dims(roi_resized / 255.0, axis=0)

            # Predict
            prediction = model.predict(roi_input, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

            if confidence >= CONFIDENCE_THRESHOLD:
                predictions_buffer.append(predicted_class)

                # Stable prediction logic
                if len(predictions_buffer) == predictions_buffer.maxlen and len(set(predictions_buffer)) == 1:
                    predicted_letter = labels_dict[predicted_class]
                    if predicted_letter != last_spoken:
                        result_text += predicted_letter
                        speak(predicted_letter)
                        last_spoken = predicted_letter
                    predictions_buffer.clear()

            # Draw info
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show prediction + confidence
            pred_text = f"{labels_dict[predicted_class]} ({confidence*100:.1f}%)"
            cv2.putText(frame, pred_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display detected text
    cv2.rectangle(frame, (0, 0), (W, 60), (255, 255, 255), -1)
    cv2.putText(frame, "Detected: " + result_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("CNN Sign Detection (Retrained)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
