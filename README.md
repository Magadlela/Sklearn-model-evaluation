# Building the Models with Scikit-learn
Contains model accuracy evaluation function

## Description
write the code to build the two models in scikit-learn. Then we’ll use k-fold cross validation to calculate the accuracy, precision,
recall and F1 score for the two models so that we can compare them.
First, we import the necessary modules and prep the data as we’ve done before.

```
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
```

### build the KFold object. 
```
kf = KFold(n_splits=5, shuffle=True)
```

three different feature matrices X1, X2 and X3. All will have the same target y
```
X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values
```


### credit
[sololearn](https://www.sololearn.com/learning/1094/3311/7382/1)
