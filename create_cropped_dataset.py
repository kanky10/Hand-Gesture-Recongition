import os
import cv2
import mediapipe as mp

# Original raw image directory (your current dataset)
RAW_DIR = './data'
# New output directory with cropped hands
CROPPED_DIR = './cropped_data'

if not os.path.exists(CROPPED_DIR):
    os.makedirs(CROPPED_DIR)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

for label in os.listdir(RAW_DIR):
    label_path = os.path.join(RAW_DIR, label)
    if not os.path.isdir(label_path):
        continue

    save_dir = os.path.join(CROPPED_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                h, w, _ = img.shape
                x_min = max(int(min(x_coords) * w) - 20, 0)
                y_min = max(int(min(y_coords) * h) - 20, 0)
                x_max = min(int(max(x_coords) * w) + 20, w)
                y_max = min(int(max(y_coords) * h) + 20, h)

                hand_crop = img[y_min:y_max, x_min:x_max]
                if hand_crop.size > 0:
                    save_path = os.path.join(save_dir, img_name)
                    cv2.imwrite(save_path, hand_crop)
