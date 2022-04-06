 
from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def train(self):
        pass