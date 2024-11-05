import src.tasks as tasks 
import src.core.actions as actions
import src.core.observation as observation

def build_env(params):
    env = tasks.available_task[params["Environment"]["task"]]( # "CombatSpider"
        **params["Environment"]["task_parameter"]
    )
    env = actions.available_action_space[params["Environment"]["action_space"]]( #"ReducedActionSpace"
        env=env
    )
    env = observation.available_observation[params["Environment"]["observation"]]( #"ImageObservation"
        env=env
    )
    return env