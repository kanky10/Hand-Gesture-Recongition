import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, classification_report
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import time

# --- Load and prepare image data for CNN ---
DATA_DIR = './data'
IMG_SIZE = 128
NUM_CLASSES = 25

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

# --- Evaluate CNN ---
cnn_model = load_model('cnn_model.h5')

start_cnn = time.time()
cnn_preds = cnn_model.predict(x_test)
cnn_time = time.time() - start_cnn

cnn_preds_classes = np.argmax(cnn_preds, axis=1)
cnn_acc = accuracy_score(y_test, cnn_preds_classes)
cnn_precision = precision_score(y_test, cnn_preds_classes, average='weighted', zero_division=0)
cnn_recall = recall_score(y_test, cnn_preds_classes, average='weighted', zero_division=0)
cnn_f1 = f1_score(y_test, cnn_preds_classes, average='weighted', zero_division=0)

# --- Load and prepare feature data for Random Forest ---
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
rf_data = np.asarray(data_dict['data'])
rf_labels = np.asarray(data_dict['labels'], dtype=int)

x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(
    rf_data, rf_labels, test_size=0.2, stratify=rf_labels
)

with open('model.p', 'rb') as f:
    rf_model = pickle.load(f)['model']

start_rf = time.time()
rf_preds = rf_model.predict(x_test_rf)
rf_time = time.time() - start_rf

rf_acc = accuracy_score(y_test_rf, rf_preds)
rf_precision = precision_score(y_test_rf, rf_preds, average='weighted', zero_division=0)
rf_recall = recall_score(y_test_rf, rf_preds, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test_rf, rf_preds, average='weighted', zero_division=0)

# --- Build comparison dataframe ---
comparison_df = pd.DataFrame({
    'Model': ['CNN', 'Random Forest'],
    'Accuracy': [cnn_acc, rf_acc],
    'Precision': [cnn_precision, rf_precision],
    'Recall': [cnn_recall, rf_recall],
    'F1 Score': [cnn_f1, rf_f1],
    'Inference Time (s)': [cnn_time, rf_time]
})

# --- Confusion Matrices ---
labels_names = [chr(i + 65) for i in range(NUM_CLASSES)]
cnn_cm = confusion_matrix(y_test, cnn_preds_classes)
rf_cm = confusion_matrix(y_test_rf, rf_preds)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cnn_cm, annot=False, fmt='d', ax=axes[0], xticklabels=labels_names, yticklabels=labels_names, cmap='Blues')
axes[0].set_title("CNN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(rf_cm, annot=False, fmt='d', ax=axes[1], xticklabels=labels_names, yticklabels=labels_names, cmap='Oranges')
axes[1].set_title("Random Forest Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
conf_matrix_path = "/mnt/data/confusion_matrices.png"
plt.savefig(conf_matrix_path)
plt.close()

# --- Save report as HTML ---
html_report_path = "/mnt/data/model_comparison_report.html"
comparison_df_html = comparison_df.to_html(index=False, float_format="%.4f")

with open(html_report_path, 'w') as f:
    f.write("<html><head><title>Model Comparison Report</title></head><body>")
    f.write("<h1>Model Performance Comparison</h1>")
    f.write(comparison_df_html)
    f.write("<h2>Confusion Matrices</h2>")
    f.write(f'<img src="confusion_matrices.png" width="800"/>')
    f.write("</body></html>")

html_report_path, conf_matrix_path
