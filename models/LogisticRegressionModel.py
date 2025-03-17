import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any
from BaseModel import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression()

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"C": 1.0, "max_iter": 100}  # 逻辑回归默认参数

    def train(self, data: list, labels: list, params: Dict[str, Any]):
        self.model.set_params(**params)
        self.model.fit(np.array(data), np.array(labels))
        print("Logistic Regression Model trained successfully.")

    def predict(self, data: list):
        return self.model.predict(np.array(data)).tolist()
