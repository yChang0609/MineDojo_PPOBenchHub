from src.core.minedojo_base import MineDojoActionBase

class DefaultActionSpace(MineDojoActionBase):
    def __init__(self, env):
        super().__init__(env)
        self.process_flow.append(f"{self.__class__.__name__}")