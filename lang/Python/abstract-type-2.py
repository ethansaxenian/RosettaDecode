from abc import ABCMeta, abstractmethod


class BaseQueue(metaclass=ABCMeta):
    """Abstract Class
    """

    def __init__(self):
        self.contents = list()

    @abstractmethod
    def Enqueue(self, item):
        pass

    @abstractmethod
    def Dequeue(self):
        pass

    def Print_Contents(self):
        for i in self.contents:
            print(i, end=' ')
