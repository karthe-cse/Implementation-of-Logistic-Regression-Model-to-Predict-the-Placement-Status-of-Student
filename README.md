# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KARTHEESHWAR D J
RegisterNumber: 21225040173
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1], 
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    
}
df = pd.DataFrame(data)
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
new_student = np.array([[6, 68, 1]])
new_student_scaled = scaler.transform(new_student)
placement_pred = model.predict(new_student_scaled)
placement_prob = model.predict_proba(new_student_scaled)
print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="1292" height="511" alt="image" src="https://github.com/user-attachments/assets/0219e74b-c534-405b-974c-26e8abc4445e" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
