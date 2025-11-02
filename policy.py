import random
from abc import ABC, abstractmethod
from typing import Literal
from models import Task, UE, BaseStation, MECServer, CloudServer

Action = Literal["local", "mec", "cloud"]

class Policy(ABC):
    @abstractmethod
    def decide(self, task: Task, ue: UE) -> Action: ...

class AlwaysLocal(Policy):
    def decide(self, task: Task, ue: UE) -> Action:
        return "local"

class AlwaysMEC(Policy):
    def decide(self, task: Task, ue: UE) -> Action:
        return "mec"

class AlwaysCloud(Policy):
    def decide(self, task: Task, ue: UE) -> Action:
        return "cloud"

class RandomPolicy(Policy):
    """Randomly selects between local, MEC, or Cloud for each task."""
    def decide(self, task: Task, ue: UE) -> Action:
        return random.choice(["local", "mec", "cloud"])

class GreedyBySize(Policy):
    """Simple professor-suggested baseline: offload large tasks."""
    def __init__(self, size_threshold_bits: float = 100e3 * 8):
        self.th = size_threshold_bits
    def decide(self, task: Task, ue: UE) -> Action:
        return "mec" if task.data_bits >= self.th else "local"
