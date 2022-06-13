# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
## Step 1:
Import the required library packages.
## Step 2:

Import the dataset to operate on.
## Step 3:

Split the dataset into required segments.
## Step 4:

Predict the required output.
## Step 5:

Run the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NITHISHWAR S
RegisterNumber:  212221230071
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("/content/spam.csv",encoding='latin-1')
dataset
dataset.head()
dataset.info()
dataset.isnull().sum()
X = dataset["v1"].values
Y = dataset["v2"].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
Y_pred
from sklearn import metrics
accuracy = metrics.accuracy_score(Y_test,Y_pred)
accuracy
```

## Output:
## Dataset:
![image](https://user-images.githubusercontent.com/94164665/173334667-7551bf8a-a777-479e-8951-0820173e7638.png)
## Dataset Head:
![image](https://user-images.githubusercontent.com/94164665/173334773-8483e459-92ef-4d03-b94c-a6a349dfed97.png)

## Dataset Information:
![image](https://user-images.githubusercontent.com/94164665/173334841-cc2b2a12-cf6f-479f-a6e0-b50cdffd13ad.png)
![image](https://user-images.githubusercontent.com/94164665/173334923-f5a69fd2-6c86-454c-a698-5074c57ee503.png)

## Predicted array:
![image](https://user-images.githubusercontent.com/94164665/173334997-e3956e18-b9ca-4095-8eab-15b44bab0876.png)
## Accuracy Score:
![image](https://user-images.githubusercontent.com/94164665/173335081-2ec47fe9-877a-41ac-8df1-45b4c28c1394.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
