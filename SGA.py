import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


df = pd.read_csv('dataset_SGA.csv')

X = df.iloc[:,1:17]
y=df.iloc[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0, test_size=0.8)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

def train_value ( input):
    classifier = SVC(kernel='rbf', C=85)
    classifier.fit(X_train, y_train)
    input=sc_X.fit_transform(input)
    

X_new = np.array([36,2,45.5,156.5,45.4,20.2,4.5,435,3.43,151.2,33.0,7.7,7.8,0.44,0.64,22]).reshape(1, -1)
train_value(X_new)