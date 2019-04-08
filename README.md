# Weather-Project-Machine-Learning-

## Intoduction
The objective of this project is to predict is Rain Tomorrow or not.

## Weather Data
Wheater Dataset contains 22 features including temperatures of the weather and Rainfall, Evaporation, Sunshine, WindSpeed, Humidity,Pressure in the air,Todays rain information.

### Try to import Pandas and Numpy to read the data.

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
%matplotlib inline
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings('ignore')

# Applied Machine Learning Models
## Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = .1)
## k-Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
## Support Vector Machine
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=.1)
## Random Forest (ensemble of Decision Trees)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state=0)
## Neural Network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier()

Identified best Model for Weather Data set using Nested cross-validation.
