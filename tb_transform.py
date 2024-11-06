from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

old_log_dir = "logs/experiment_old/"
new_log_dir = "logs/experiment_new/"

ea = event_accumulator.EventAccumulator(old_log_dir)
ea.Reload()

writer = SummaryWriter(new_log_dir)

scalar_tags = ea.Tags()["scalars"]
print("All scalar tags:", scalar_tags)

for tag in scalar_tags:
    scalar_events = ea.Scalars(tag)  
    print(f"Transform data for tag: {tag}")
    transform_tag = tag
    if tag == "Episode/Steps":
        transform_tag = "sample/MineDojo/Combat_Spider_episode_steps"
    elif tag == "Episode/Reward":
        transform_tag = "sample/MineDojo/Combat_Spider_reward"
    for event in scalar_events:
        writer.add_scalar(transform_tag, event.value, event.step)

writer.close()