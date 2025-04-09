import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Init MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Labels
labels_dict = {i: chr(65 + i) for i in range(25)}

# Tracking prediction history for "live accuracy"
prediction_history = deque(maxlen=30)
consistent_prediction = None
frame_count = 0

# Start video
cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x, y = lm.x, lm.y
                x_.append(x)
                y_.append(y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # Prediction
        prediction = model.predict([np.asarray(data_aux)])[0]
        probabilities = model.predict_proba([np.asarray(data_aux)])[0]
        confidence = np.max(probabilities)

        predicted_char = labels_dict[int(prediction)]
        prediction_history.append(predicted_char)

        # Display prediction and confidence
        cv2.putText(frame, f"Predicted: {predicted_char}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Estimate live "accuracy" based on prediction consistency
        most_common = max(set(prediction_history), key=prediction_history.count)
        consistency = prediction_history.count(most_common) / len(prediction_history)
        cv2.putText(frame, f"Live Accuracy (consistency): {consistency*100:.1f}%", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 180, 0), 2)

    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow('Random Forest - Live Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
