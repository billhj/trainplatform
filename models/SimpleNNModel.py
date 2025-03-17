from BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNNModel(BaseModel):
    def __init__(self):
        self.input_features = 10
        self.middle_features = 5
        self.output_features = 1
        self.lr = 0.01
        self.epochs = 100
        self.model = nn.Sequential(
            nn.Linear(self.input_features, self.middle_features),
            nn.ReLU(),
            nn.Linear(self.middle_features, self.output_features),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def define_model(self, input_features, middle_features, output_features, lr = 0.01, epochs = 100):
        self.input_features = input_features
        self.middle_features = middle_features
        self.output_features = output_features
        self.lr = lr
        self.epochs = epochs
        self.model = nn.Sequential(
            nn.Linear(self.input_features, self.middle_features),
            nn.ReLU(),
            nn.Linear(self.middle_features, self.output_features),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    @property
    def default_params(self) -> dict[str, object]:
        return {"epochs": self.epochs, "learning_rate": self.lr}

    def train(self, data: list, labels: list, params: dict[str, object]):
        if "learning_rate" in params:
            self.lr = params.get("learning_rate")
        if "epochs" in params:
            self.epochs = params.get("epochs")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_train = torch.tensor(data, dtype=torch.float32)
        y_train = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch [{self.epochs + 1}/{self.epochs}], Loss: {loss.item()}")

    def predict(self, data: list):
        X_test = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_test).numpy().tolist()
        return predictions