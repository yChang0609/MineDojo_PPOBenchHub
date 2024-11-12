from gym import Wrapper
from src.core.minedojo_base import MineDojoActionBase

class DefaultActionSpace(MineDojoActionBase):
    def __init__(self, env):
        super().__init__(env)
        self.process_flow.append(f"{self.__class__.__name__}")
    def action(self, action):
        return action
    

class ObsPassingEnv(Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs
    
class DefaultWithActionMaskActionSpace(MineDojoActionBase):
    def __init__(self, env):
        env = ObsPassingEnv(env)
        super().__init__(env)
        self.process_flow.append(f"{self.__class__.__name__}")
        

    def action(self, action):
        action_mask = self.env.current_obs.get('masks', None)
        return action if self._valid_action(action, action_mask) else self.env.action_space.no_op()
    
    def _valid_action(self, action, mask):
        ret = False
        if (action[5] > 3):
            if(mask["action_type"][action[5]]):
                if(action[5] == 4 ): # functional actions 'craft'
                    if(mask["craft_smelt"][action[6]]):
                        ret = True
                elif(action[5] == 5): # functional actions 'equip'
                    if(mask["equip"][action[7]]):
                        ret = True
                elif(action[5] == 6): # functional actions 'place'
                    if(mask["place"][action[7]]):
                        ret = True
                elif(action[5] == 7): # functional actions 'destroy'
                    if(mask["destroy"][action[7]]):
                        ret = True
        else:
            ret = True

        return ret