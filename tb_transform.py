import os
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

def data_refactor(
        source_dir,
        target_path,
        refactor_label:dict,
        refactor_wight:dict
    ):
    dir = os.path.split(source_dir)[-1]
    target_dir = os.path.join(target_path, dir) + "_refactor"

    ea = event_accumulator.EventAccumulator(source_dir + "/PPO/")
    ea.Reload()
    print(ea.path)
    writer = SummaryWriter(target_dir + "/")

    scalar_tags = ea.Tags()["scalars"]
    print("All scalar tags:", scalar_tags)

    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)  
        print(f"Transform data for tag: {tag}")
        transform_tag = tag
        wigth = 1
        if  tag in refactor_label.keys():
            transform_tag = refactor_label[tag]
            wigth = refactor_wight[tag]
        for event in scalar_events:
            writer.add_scalar(transform_tag, event.value*wigth, event.step)
    writer.close()

task_name = "CombatSpider"
refactor_label = {
    "Episode/Env_0/Episode_Steps":f"sample/{task_name}_episode_steps",
    "Episode/Env_0/Episode_Reward":f"sample/{task_name}_reward",
}

refactor_wight = {
    "Episode/Env_0/Episode_Steps": 1/4,
    "Episode/Env_0/Episode_Reward": 1.0,
}

# old_log_dir = "runs/HuntCow-1M-MLP_PPO" #Note: format: "path to folder/folder", not use "path to folder/folder[ / ]"!!
path = "logs/ppo_CombatSpider" #Note: format: "path to folder/folder", not use "path to folder/folder[ / ]"!!

folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
for f in folders:
   data_refactor(
       os.path.join(path, f),
       path+"_refactor",
       refactor_label,
       refactor_wight
       )