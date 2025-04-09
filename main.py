import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyttsx3  # Importing the text-to-speech library

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# Initialize variables
cap = cv2.VideoCapture(0)
result_text = ""
predictions_buffer = deque(maxlen=5)  # Buffer to hold last 5 predictions
frame_counter = 0  # Frame counter to control inference frequency

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Process the frame only every 2nd frame
        frame_counter += 1
        if frame_counter % 2 == 0:  # Process every 2nd frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                prediction = model.predict([np.asarray(data_aux)])

                # Add the prediction to the buffer
                predictions_buffer.append(prediction[0])

                # Check if all predictions in the buffer are the same
                if len(predictions_buffer) == predictions_buffer.maxlen and len(set(predictions_buffer)) == 1:
                    predicted_character = labels_dict[int(prediction[0])]
                    result_text += predicted_character
                    predictions_buffer.clear()  # Clear the buffer after adding the character

                    # Speak the predicted character
                    engine.say(predicted_character)  # Convert text to speech
                    engine.runAndWait()  # Wait until speaking is finished

    # Draw result text as a dashboard
    cv2.rectangle(frame, (0, 0), (W, 60), (245, 245, 245), -1)
    cv2.putText(frame, "Detected: " + result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the frame with the detected gestures and dashboard
    cv2.imshow('Sign Language Dashboard', frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
