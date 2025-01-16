import src.tasks as tasks 
import src.core.actions as actions
import src.core.observation as observation

import pprint
import yaml

import numpy as np

def build_env(params, seed=None):
    env = tasks.available_task[params["Environment"]["task"]]( 
        **params["Environment"]["task_parameter"]
    )
    env = actions.available_action_space[params["Environment"]["action_space"]]( #"ReducedActionSpace"
        env=env
    )
    env = observation.available_observation[params["Environment"]["observation"]]( #"ImageObservation"
        env=env
    )
    # seed setting priority [function input seed] -> [params seed] -> [np random seed]
    seed = seed or params["Environment"]["seed"] or np.random.randint(0, 10000)
    env.seed(seed)
    print(f"Env seed : {seed}")
    return env



def load_config(config_path):
    params = None
    with open(config_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    return params