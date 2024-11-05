from src.core.minedojo_base import MineDojoObservationBase
from gym.spaces import Box
import numpy as np

class ImageObservation(MineDojoObservationBase):
    def __init__(self, env):
        super().__init__(env)
        self.process_flow.append(f"{self.__class__.__name__}")
        observation_space = self.env.observation_space['rgb']
        self.observation_space = observation_space

    def observation(self, obs):
        # Get the RGB observation
        return obs['rgb']
    

