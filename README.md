# Diabetes-Prediction-using-SVM-Machine-Learning-Project-
This project uses Support Vector Machine (SVM) with a Linear Kernel to predict whether a person is diabetic based on medical attributes from the PIMA Diabetes Dataset. The model is trained using scikit-learn and includes data preprocessing, model training, evaluation, and a simple predictive system for custom inputs

📂 Project Structure
├── diabetes.csv        # Dataset
├── diabetes_predict.py # Main ML code
└── README.md           # Documentation

📊 Dataset
The dataset used is the PIMA Indians Diabetes Database, which contains medical diagnostic measurements such as:
Pregnancies
Glucose level
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
Outcome (0 = Not Diabetic, 1 = Diabetic)

🚀 Features
✔ Loads and processes the dataset
✔ Standardizes feature values
✔ Splits dataset into training & testing sets
✔ Trains an SVM model
✔ Evaluates accuracy
✔ Provides a prediction system for new input data

🧠 Technologies Used
Python
NumPy
Pandas
scikit-learn

📌 Notes
The dataset is imbalanced, so accuracy may vary.
You can improve performance using:
Hyperparameter tuning
Different kernels
Feature engineering
StandardScaler

SVM Classifier (sklearn.svm.SVC)
