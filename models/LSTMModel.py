from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models.BaseModel import BaseModel
from typing import Dict, Any
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        LSTM 模型初始化。

        参数:
        - input_size: 输入特征的维度（例如，时间序列的每个时间步的特征数）。
        - hidden_size: 隐藏层的维度。
        - num_layers: LSTM 的层数。
        - output_size: 输出的维度（例如，分类的类别数或回归的目标值）。
        - dropout: Dropout 概率，用于防止过拟合。
        """
        super(LSTM, self).__init__()

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,  # 指定 LSTM 的层数
            batch_first=True,  # 输入数据的形状为 (batch_size, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0  # 仅在多层 LSTM 时使用 dropout
        )

        # 全连接层，将 LSTM 的输出映射到目标输出维度
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播。

        参数:
        - x: 输入数据，形状为 (batch_size, seq_len, input_size)。

        返回:
        - output: 模型输出，形状为 (batch_size, output_size)。
        """
        # LSTM 的输出包含两个部分：
        # 1. output: 每个时间步的隐藏状态，形状为 (batch_size, seq_len, hidden_size)
        # 2. (h_n, c_n): 最后一个时间步的隐藏状态和细胞状态
        output, (h_n, c_n) = self.lstm(x)

        # 只取最后一个时间步的隐藏状态
        last_hidden_state = output[:, -1, :]

        # 应用 dropout
        last_hidden_state = self.dropout(last_hidden_state)

        # 通过全连接层得到最终输出
        output = self.fc(last_hidden_state)

        return output


class LSTMModel(BaseModel):
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 100
        self.input_size = 10
        self.hidden_size = 10
        self.num_layers = 1
        self.output_size = 1
        self.dropout = 0.2
        self.criterion = nn.MSELoss()  # nn.CrossEntropyLoss()#nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 32
        self.model = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          output_size=self.output_size, dropout=self.dropout)

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"epochs": self.epochs, "learning_rate": self.learning_rate, "input_size": self.input_size, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "output_size": self.output_size, "dropout": self.dropout, "batch_size": self.batch_size}

    def train(self, data: list, labels: list, params: Dict[str, Any]):
        if "learning_rate" in params:
            self.learning_rate = params.get("learning_rate")
        if "epochs" in params:
            self.epochs = params.get("epochs")
        if "input_size" in params:
            self.input_size = params.get("input_size")
        if "hidden_size" in params:
            self.hidden_size = params.get("hidden_size")
        if "num_layers" in params:
            self.num_layers = params.get("num_layers")
        if "output_size" in params:
            self.output_size = params.get("output_size")
        if "dropout" in params:
            self.dropout = params.get("dropout")
        if "batch_size" in params:
            self.batch_size = params.get("batch_size")

        self.model = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, output_size=self.output_size, dropout=self.dropout)
        self.criterion = nn.MSELoss()  # nn.CrossEntropyLoss()#nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        # 转换为 PyTorch 的 Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        # 创建 DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        """
            训练 LSTM 模型。

            参数:
            - model: LSTM 模型实例。
            - train_loader: 训练集的 DataLoader。
            - test_loader: 测试集的 DataLoader。
            - criterion: 损失函数。
            - optimizer: 优化器。
            - num_epochs: 训练轮数。
            """
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            # 训练阶段
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # 计算平均训练损失
            train_loss /= len(train_loader)

            # 验证阶段
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    test_loss += loss.item()

            # 计算平均测试损失
            test_loss /= len(test_loader)
            # 打印训练和测试损失
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    def predict(self, data: list):
        outputs = self.model(data)
        return outputs