from src.core.minedojo_base import MineDojoObservationBase

class DefaultObservation(MineDojoObservationBase):
    def __init__(self, env):
        super().__init__(env)
        self.process_flow.append(f"{self.__class__.__name__}")
    def observation(self, obs):
        return obs