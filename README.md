# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

5.Generate Confusion Matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: RAMYA S 
RegisterNumber: 212222040130  
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris = load_iris()
df = pd.DataFrame(data = iris.data , columns = iris.feature_names)
df['target'] = iris.target
print(df.head())
x = df.drop('target',axis=1)
y = df['target']
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42)
sgd_clf = SGDClassifier(max_iter = 1000 , tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.3f}")
cm = confusion_matrix(y_test,y_pred)
print("confusion Matrix:")
print(cm)

```

## Output:
![image](https://github.com/user-attachments/assets/ad23f5b7-3ce9-44f4-8407-015200d79124)

![image](https://github.com/user-attachments/assets/a0aabb22-da49-4140-bcc9-6eb5884dbc5f)

![image](https://github.com/user-attachments/assets/1fc03ced-73f9-4ff8-b6bd-9c13613f6d00)

![image](https://github.com/user-attachments/assets/a8639b78-f335-424b-a5b3-40140fecfd31)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
