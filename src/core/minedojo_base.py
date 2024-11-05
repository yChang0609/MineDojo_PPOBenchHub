from gym import Wrapper, ObservationWrapper, ActionWrapper

class MineDojoBase():
    def __init__(
        self,
        env
    ):
        self._env_base = env
        self.process_flow = []
    def __str__(self):
        our_str =  f"{self.process_flow} "
        if isinstance(self._env_base, MineDojoBase):
            our_str += f"<- {self._env_base}"
        return  our_str
    
class MineDojoEnvdBase(Wrapper, MineDojoBase):
    def __init__(
        self, env
    ):
        Wrapper.__init__(self, env)  
        MineDojoBase.__init__(self, env)
        

class MineDojoObservationBase(ObservationWrapper, MineDojoBase):
    def __init__(
        self, env
    ):
        Wrapper.__init__(self, env)  
        MineDojoBase.__init__(self, env)  
    
class MineDojoActionBase(ActionWrapper, MineDojoBase):
    def __init__(
        self, env
    ):
        Wrapper.__init__(self, env)  
        MineDojoBase.__init__(self, env)  