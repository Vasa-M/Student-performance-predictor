
# 🎓 Student Performance Predictor

This is a machine learning web application built using **Streamlit** that predicts whether a student will **pass or fail** based on their study time, absences, and past failures. The model is powered by a **Random Forest Classifier** with hyperparameter tuning to improve accuracy.

---

## 📌 Features

- Predicts student outcome (pass/fail) using 3 key features
- Trained using Random Forest Classifier
- Hyperparameter tuning with RandomizedSearchCV
- Confusion matrix visualization using Seaborn
- Simple and interactive Streamlit web interface

---

## 🧠 Model Details

- Model: Random Forest Classifier  
- Accuracy: ~**89.98%**
- Tuned Accuracy: ~**87.37%**
- Cross-Validation Accuracy: ~**83.51%**

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vasa-M/Student-performance-predictor.git
cd student-performance-predictor
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📊 How to Use

1. Launch the Streamlit app.
2. Adjust the sliders for:
   - **Study Time** (hours/week)
   - **Absences** (days)
   - **Failures** (subjects)
3. The app will display whether the student is likely to **pass or fail**.
4. The confusion matrix gives an overview of the model’s predictions.

---

## 🧾 Dataset

- File: `student-performance.csv`
- Columns used: `studytime`, `absences`, `failures`, and final grade (`G3`)
- A new column `pass_fail` is created where `G3 >= 10` is **Pass**, otherwise **Fail**.
- Source: UCI Machine Learning Repository

---


## 📬 Contact

For any inquiries or suggestions, feel free to reach out to me at [vasanthmv27@gmail.com].

---
