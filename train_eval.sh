# CONFIG_FILE=configs/100K_CNN_CombatSpider_default-ac.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/100K_MLP_CombatSpider_default-ac.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/1M_CNN_CombatSpider_default-ac.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/1M_MLP_CombatSpider_default-ac.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/100K_CNN_CombatSpider.yaml
# # MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/100K_MLP_CombatSpider.yaml
# # MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/1M_CNN_CombatSpider.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/1M_MLP_CombatSpider.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/hunt_cow/100K_MLP_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/hunt_cow/100K_CNN_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/hunt_cow/1M_MLP_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE

# CONFIG_FILE=configs/hunt_cow/1M_CNN_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
# MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE



# MINEDOJO_HEADLESS=1 python train.py --config experiments/hunt_cow/100K_CNN_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python train.py --config configs/hunt_cow/100K_CLIP_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python train.py --config experiments/hunt_cow/100K_MLP_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python -u train.py --config configs/100K_CLIP_CombatSpider.yaml

# MINEDOJO_HEADLESS=1 python -u train.py --config experiments/1M_CLIP_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python -u train.py --config experiments/1M_CNN_HuntCow.yaml
# MINEDOJO_HEADLESS=1 python -u eval.py --config experiments/1M_CLIP_HuntCow.yaml

# MINEDOJO_HEADLESS=1 python -u eval.py --config experiments/combat_spider/1M_CLIP_CombatSpider.yaml
# MINEDOJO_HEADLESS=1 python -u eval.py --config experiments/combat_spider/1M_CNN_CombatSpider.yaml
# "logs/HuntCowEasy/1M_CLIP_HuntCow/PPO" \
# "logs/HuntCowEasy/100K_CLIP_HuntCow/PPO" \

# MINEDOJO_HEADLESS=1 python -u eval.py -logs \
#     "logs/HuntCowEasy/1M_CLIP_HuntCow-OnlyCow/PPO" \
#     "logs/HuntCowEasy/100K_CLIP_HuntCow-OnlyCow/PPO" 

# MINEDOJO_HEADLESS=1 python -u train.py --config configs/1M_CLIP_HarvestMilk.yaml
# MINEDOJO_HEADLESS=1 python -u train.py --config configs/hunt_cows/100K_CLIP_HuntCows.yaml

# MINEDOJO_HEADLESS=1 python -u train.py --config experiments/env_test/100K_CLIP_HuntCows_q3.yaml
# MINEDOJO_HEADLESS=1 python -u train.py --config experiments/env_test/100K_CLIP_HuntCows_q5.yaml
# MINEDOJO_HEADLESS=1 python -u train.py --config configs/1M_CLIP_HarvestWool.yaml
MINEDOJO_HEADLESS=1 python -u train.py --config configs/1M_CLIP_HarvestMilk.yaml


# MINEDOJO_HEADLESS=1 python -u train.py --config experiments/env_test/100K_CLIP_HuntCows_q10.yaml

# MINEDOJO_HEADLESS=1 python -u train.py --config configs/hunt_cows/100K_CLIP_HuntCows_10.yaml
# MINEDOJO_HEADLESS=1 python -u train.py --config configs/hunt_cows/100K_CLIP_HuntCows_20.yaml

# MINEDOJO_HEADLESS=1 python -u eval.py -logs \
#     "logs/ppo_HuntCows/100K_CLIP_HuntCows_3/PPO_1" \
#     "logs/ppo_HuntCows/100K_CLIP_HuntCows_10/PPO_1" \
#     "logs/ppo_HuntCows/100K_CLIP_HuntCows_20/PPO_1" \

# MINEDOJO_HEADLESS=1 python -u eval.py -logs \
#     "logs/ppo_HarvestMilk/1M_CLIP_HarvestMilk/PPO_1" 