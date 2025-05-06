# student_performance_predictor.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_and_train_model():
    # Load the dataset
    df = pd.read_csv("assets/student-performance.csv")

    # Create 'pass_fail' column
    df['pass_fail'] = df['G3'].apply(lambda grade: 'pass' if grade >= 10 else 'fail')

    # Select features and target
    X = df[['studytime', 'absences', 'failures']]
    y = df['pass_fail'].map({'fail': 0, 'pass': 1})

    # Initialize Random Forest model
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model_rf.fit(X, y)

    # Hyperparameter Tuning with RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
    random_search.fit(X, y)

    best_model = random_search.best_estimator_

    # Get predictions on the training data with the best model
    y_pred_best = best_model.predict(X)

    # Calculate accuracy
    accuracy_best = accuracy_score(y, y_pred_best)

    # Create confusion matrix
    cm = confusion_matrix(y, y_pred_best)

    return best_model, accuracy_best, cm

def predict(model, user_input):
    # Use the trained model to predict on user input
    return model.predict(user_input)

def plot_confusion_matrix(cm):
    # Plot confusion matrix with scaled colors
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'], vmin=0, vmax=cm.max())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    return plt

