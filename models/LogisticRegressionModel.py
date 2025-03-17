from BaseModel import BaseModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression()
        self.C = 1.0
        self.max_iter = 100

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"C": self.C, "max_iter": self.max_iter}  # 逻辑回归默认参数

    def train(self, data: list, labels: list, params: Dict[str, Any]):
        if "C" in params:
            self.C = params.get("C")
        if "max_iter" in params:
            self.max_iter = params.get("max_iter")
        self.model.set_params(**params)
        self.model.fit(np.array(data), np.array(labels))
        print("Logistic Regression Model trained successfully.")

    def predict(self, data: list):
        return self.model.predict(np.array(data)).tolist()

"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegressionModel()
model.train(X_train, y_train, params={'C': 1.0, 'max_iter': 200})
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
"""

