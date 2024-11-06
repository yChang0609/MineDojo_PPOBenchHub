import src.tasks as tasks 
import src.core.actions as actions
import src.core.observation as observation

import numpy as np

def build_env(params, seed=None):
    env = tasks.available_task[params["Environment"]["task"]]( # "CombatSpider"
        **params["Environment"]["task_parameter"]
    )
    env = actions.available_action_space[params["Environment"]["action_space"]]( #"ReducedActionSpace"
        env=env
    )
    env = observation.available_observation[params["Environment"]["observation"]]( #"ImageObservation"
        env=env
    )
    # seed setting priority [function input seed] -> [params seed] -> [np random seed]
    if seed == None and params["Environment"]["seed"] == None:
        seed = np.random.randint(0, 10000)
    else:
        seed = params["Environment"]["seed"]
    env.seed(seed)
    print(f"Env seed : {seed}")
    return env