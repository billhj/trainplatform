from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def train(self, data: list, labels: list, params: Dict[str, Any]):
        pass

    @abstractmethod
    def predict(self, data: list):
        pass