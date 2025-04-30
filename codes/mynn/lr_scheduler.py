from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        # 调用父类的初始化方法
        super().__init__(optimizer)
        # 学习率衰减的里程碑步数列表
        self.milestones = milestones
        # 学习率衰减的系数
        self.gamma = gamma

    def step(self) -> None:
        # 步数加1
        self.step_count += 1
        # 检查当前步数是否达到里程碑
        if self.step_count in self.milestones:
            # 衰减学习率
            self.optimizer.init_lr *= self.gamma

class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.9) -> None:
        # 调用父类的初始化方法
        super().__init__(optimizer)
        # 学习率衰减的系数
        self.gamma = gamma

    def step(self) -> None:
        # 步数加1
        self.step_count += 1
        # 按照指数衰减公式更新学习率
        self.optimizer.init_lr *= self.gamma