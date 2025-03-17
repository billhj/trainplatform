from BaseModel import BaseModel
import numpy as np

class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.lr = 0.01
        self.epochs = 100
        self.batch_size = 32

    @property
    def default_params(self) -> dict[str, object]:
        return {
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

    def train(self, data: list, labels: list, params: dict[str, object]):
        # 提取参数
        if "learning_rate" in params:
            self.lr = params.get("learning_rate")
        if "epochs" in params:
            self.epochs = params.get("epochs")
        if "batch_size" in params:
            self.batch_size = params.get("batch_size")

        # 转换数据为 numpy 数组
        X = np.array(data).reshape(-1, len(data[0]))
        y = np.array(labels)

        # 初始化权重和偏置
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 训练模型
        for _ in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                batch_y_pred = np.dot(batch_X, self.weights) + self.bias
                errors = batch_y - batch_y_pred

                # 更新权重和偏置
                self.weights += self.lr * (2 / self.batch_size) * np.dot(batch_X.T, errors)
                self.bias += self.lr * (2 / self.batch_size) * np.sum(errors)

        print(f"Training completed with {params}")

    def predict(self, data: list):
        X = np.array(data).reshape(-1, len(data[0]))
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred.tolist()