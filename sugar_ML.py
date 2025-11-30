import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset (your path)
diabetes_dataset = pd.read_csv(r"C:\Users\Suhas\OneDrive\Desktop\MIT\Guide\diabetes.csv")

# Separate features and label
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy
train_accuracy = accuracy_score(Y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(Y_test, classifier.predict(X_test))

# Predictive system
input_data = (1,89,66,23,94,28.1,0.167,21)
input_data_np = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_np)
prediction = classifier.predict(std_data)[0]

# Final output only
print("\nFINAL OUTPUT:")
print("Training Accuracy :", train_accuracy)
print("Test Accuracy     :", test_accuracy)
print("Input Prediction  :", "Diabetic" if prediction == 1 else "Not Diabetic")
