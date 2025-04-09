# compare_models.py
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Setup
DATA_DIR = './data'
IMG_SIZE = 128
NUM_CLASSES = 25

# Load data
data = []
labels = []

for label in os.listdir(DATA_DIR):
    for img_name in os.listdir(os.path.join(DATA_DIR, label)):
        img_path = os.path.join(DATA_DIR, label, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(int(label))

X = np.array(data) / 255.0
y = np.array(labels)
y_cat = to_categorical(y, NUM_CLASSES)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
_, _, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, stratify=y)

# --- Evaluate CNN ---
cnn_model = load_model('cnn_model.h5')
cnn_preds = cnn_model.predict(x_test)
cnn_preds_classes = np.argmax(cnn_preds, axis=1)
cnn_acc = accuracy_score(y_test, cnn_preds_classes)
print(f"CNN Accuracy: {cnn_acc * 100:.2f}%")

# --- Evaluate Random Forest ---
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
rf_data = np.asarray(data_dict['data'])
rf_labels = np.asarray(data_dict['labels'], dtype=int)

x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(
    rf_data, rf_labels, test_size=0.2, stratify=rf_labels
)

with open('model.p', 'rb') as f:
    rf_model = pickle.load(f)['model']

rf_preds = rf_model.predict(x_test_rf)
rf_acc = accuracy_score(y_test_rf, rf_preds)
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")

# --- Plot Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cnn_cm = confusion_matrix(y_test, cnn_preds_classes)
rf_cm = confusion_matrix(y_test_rf, rf_preds)

labels = [chr(i + 65) for i in range(NUM_CLASSES)]

disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cnn_cm, display_labels=labels)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=labels)

disp_cnn.plot(ax=axes[0], cmap='Blues', xticks_rotation=45)
axes[0].set_title("CNN Confusion Matrix")

disp_rf.plot(ax=axes[1], cmap='Oranges', xticks_rotation=45)
axes[1].set_title("Random Forest Confusion Matrix")

plt.tight_layout()
plt.show()
