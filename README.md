# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. mport the standard libraries.
  
2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values. 

## Program:
```
*/
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:A.S.BHARANIKA
RegisterNumber:212224040048
*/
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv("Salary_EX7.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Position"] = le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

plt.figure(figsize=(20, 8))

plot_tree(dt, feature_names=x.columns, filled=True)

plt.show()

```

## Output:
![Screenshot 2025-04-20 092957](https://github.com/user-attachments/assets/35ccfb15-5b0c-46f9-a5d6-30b8f0e99614)
![Screenshot 2025-04-20 093110](https://github.com/user-attachments/assets/da1d91b8-5334-4bd1-9115-3d5f18763a9a)
![Screenshot 2025-04-20 093219](https://github.com/user-attachments/assets/635bc89c-9ab6-4f22-829a-19274a7cd2b1)





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
