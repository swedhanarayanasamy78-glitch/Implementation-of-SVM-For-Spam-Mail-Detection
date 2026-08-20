# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect and label email data as spam or non-spam (ham).

2.Preprocess text using tokenization, stop-word removal, and vectorization (e.g., TF-IDF).

3.Split dataset into training and testing sets.

4.Train the SVM classifier to find the optimal hyperplane separating spam and ham.

5.Test the model and use it to predict whether new emails are spam or not.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:SWEDHA.N 
RegisterNumber:  212225230280
*/
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (Spam dataset with 'label' and 'message' columns)
data = pd.read_csv(r"C:\Users\acer\Downloads\spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels: spam = 1, ham = 0
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with custom message
sample = ["Congratulations! You won a free lottery ticket"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

if prediction[0] == 1:
    print("\nSpam Message")
else:
    print("\nNot Spam Message")

```

## Output:
<img width="567" height="320" alt="image" src="https://github.com/user-attachments/assets/a9db87db-1a35-4874-a6af-3d278b356bf8" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
