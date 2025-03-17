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
        self.C = params.get("C")
        self.max_iter = params.get("max_iter")
        self.model.set_params(**params)
        self.model.fit(np.array(data), np.array(labels))
        print("Logistic Regression Model trained successfully.")

    def predict(self, data: list):
        return self.model.predict(np.array(data)).tolist()
