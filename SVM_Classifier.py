import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# 1. Setting folder paths and image size
base_path = 'C:/Users/Atharva/PycharmProjects/SkillCraft/TASK 3'
data_folder = os.path.join(base_path, 'data/train')
model_folder = os.path.join(base_path, 'models')
results_folder = os.path.join(base_path, 'results')
image_size = (64, 64)

# Ensuring the output folders exist
os.makedirs(model_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# 2. Load and preprocess images
X = []
y = []
filenames = []

for file in os.listdir(data_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(data_folder, file)
        img = cv2.imread(path)
        img = cv2.resize(img, image_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(gray.flatten())  # Convert 2D to 1D
        filenames.append(file)
        y.append(0 if 'cat' in file.lower() else 1)  # 0 = Cat, 1 = Dog

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
    X_scaled, y, filenames, test_size=0.2, random_state=42
)

# 5. Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 6. Make predictions and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# 7. Save trained model
model_path = os.path.join(model_folder, 'svm_model.pkl')
joblib.dump(model, model_path)

# 8. Save prediction results to CSV
results = pd.DataFrame({
    'Image': f_test,
    'Prediction': ['Cat' if pred == 0 else 'Dog' for pred in y_pred]
})

csv_path = os.path.join(results_folder, 'predictions.csv')
results.to_csv(csv_path, index=False)

print("✅ Model and predictions saved successfully.")
