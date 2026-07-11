import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 

from glassboxml.models.decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


"This experiment tests the performance of my decision tree model against the one built by scikit-learn."
  
"python -m experiments.decision_tree.experiment3"

# fetch dataset 
# wine_quality = fetch_ucirepo(id=186) 

wine_quality = pd.read_csv("experiments/data/wine_quality.csv")

model = DecisionTree(max_depth=8)

X = wine_quality.iloc[:, :-1]
y = wine_quality.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

pred = model.predict_sample(X_test.iloc[0].values, model.root)

accuracy = np.mean(pred == y_test)
print(accuracy)


model2 = DecisionTreeClassifier(max_depth=8, random_state=42)
model2.fit(X_train, y_train)
predictions = model2.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(accuracy)