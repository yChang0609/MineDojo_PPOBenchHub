from minedojo.sim.spaces import MultiDiscrete
# from gym.spaces import MultiDiscrete
from src.core.minedojo_base import MineDojoActionBase

class ReducedActionSpace(MineDojoActionBase):
    def __init__(self, env):
        super().__init__(env)
        self.process_flow.append(f"{self.__class__.__name__}")
        self.action_space = MultiDiscrete([12, 3], noop_vec=[0, 0])
        self.action_map = {
            1: [1, 0, 0],  # forward
            2: [2, 0, 0],  # backward
            3: [0, 1, 0],  # left
            4: [0, 2, 0],  # right
            5: [0, 0, 1],  # jump
            6: [0, 0, 2],  # sneak
            7: [0, 0, 3],  # sprint
            8: [0, 0, 0, 11, 12],  # camera pitch +30
            9: [0, 0, 0, 13, 12],  # camera pitch -30
            10: [0, 0, 0, 12, 11],  # camera yaw +30
            11: [0, 0, 0, 12, 13]  # camera yaw -30
        }
    def action(self, action):
        action_t = self.env.action_space.no_op() # [0, 0, 0, 12, 12, 0, 0, 0]
        # process action 0
        if action[0] in self.action_map:
            mapped_values = self.action_map[action[0]]
            for i in range(len(mapped_values)):
                action_t[i] = mapped_values[i]
        # process action 1
        if action[1] == 1:
            action_t[5] = 1  # use
        elif action[1] == 2:
            action_t[5] = 3  # attack
        return action_t
