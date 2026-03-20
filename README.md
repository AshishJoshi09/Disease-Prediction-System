 🩺 Disease Prediction System (Diabetes & Heart Disease)

 📌 Overview

This project is a **Machine Learning-based Disease Prediction System** that predicts the likelihood of **Diabetes** and **Heart Disease** using patient health data.

It uses **Logistic Regression** to classify whether a person is likely to have a disease based on medical attributes.

🚀 Features

* 📂 Supports multiple datasets (Diabetes & Heart Disease)
* 🧹 Data preprocessing:

  * Handles missing values using mean imputation
  * Converts categorical variables using one-hot encoding
    
* ⚖️ Feature scaling using StandardScaler
* 🤖 Machine learning model: Logistic Regression
* 📊 Model evaluation:
  * Accuracy Score
  * Classification Report
  * Confusion Matrix
  * ROC-AUC Score
    
* 🔮 Predicts disease outcome for new user input


🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn

 ⚙️ How It Works

1. Load dataset (e.g., diabetes or heart disease dataset)
2. Preprocess data:

   * Handle missing values
   * Encode categorical variables
3. Split dataset into training and testing sets
4. Scale features for better performance
5. Train Logistic Regression model
6. Evaluate model performance
7. Predict disease outcome for new input

 ▶️ How to Run

1. Clone the Repository

git clone https://github.com/AshishJoshi09/Disease-Prediction-System.git
cd Disease-Prediction-System

2. Install Dependencies

pip install pandas scikit-learn

3. Run the Program

python Disease3.py

📥 Input Instructions

* Enter dataset filename (e.g., `diabetes.csv` or `heart.csv`)
* Enter target column (e.g., `Outcome` or `HeartDisease`)

📊 Output

* Accuracy Score
* Classification Report
* Confusion Matrix
* ROC-AUC Score
* Prediction result (Positive / Negative)

 🎯 Use Cases

* Early disease prediction (educational purpose)
* Learning machine learning workflow
* Practicing healthcare data analysis


⚠️ Disclaimer

This project is for **educational purposes only** and **not intended for medical diagnosis**.

 👨‍💻 Author

Ashish Joshi


