import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)

x = train.drop(columns='class', axis=1)
y = train['class']
test_x = test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=5)
y_train = y_train.values.ravel()

model = RandomForestClassifier(random_state=0)

clf=model.fit(x_train, y_train)
print(model.score(x_test, y_test))


y_pred = np.argmax(model.predict_proba(test_x), axis=1)
print(y_pred)

submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)

