# app.py
import streamlit as st
from student_performance_predictor import load_and_train_model, predict, plot_confusion_matrix

# Load and train the model (in the background)
model, accuracy, cm = load_and_train_model()

# Show model accuracy
st.write(f"🎯 Best Model Accuracy: {accuracy * 100:.2f}%")

# Show confusion matrix
print(cm)
st.pyplot(plot_confusion_matrix(cm))

# Streamlit app layout
st.title("Student Performance Prediction")
studytime = st.slider("Study Time (hours/week)", 0, 20, 10)
absences = st.slider("Absences (days)", 0, 30, 10)
failures = st.slider("Failures (subjects)", 0, 5, 0)

# Make prediction based on user input
user_input = [[studytime, absences, failures]]
prediction = predict(model, user_input)

# Display the result
if prediction == 1:
    st.success("The student will pass!")
else:
    st.error("The student will fail!")
