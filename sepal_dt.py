import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def sdt(a):
    split=a
    df=load_iris()
    x=df.data[:,0:2]
    y=df.target
    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(x,y, train_size=split, random_state=1)

    dtc = DecisionTreeClassifier()
    dtc.fit(train_inputs, train_classes)
    y_pred = dtc.predict(test_inputs)
    accuracy=accuracy_score(test_classes ,y_pred)
   # print(accuracy)
    return accuracy
#spdt()
