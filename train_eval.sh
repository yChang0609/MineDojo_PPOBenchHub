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



MINEDOJO_HEADLESS=1 python train.py --config experiments/hunt_cow/100K_CNN_HuntCow.yaml
MINEDOJO_HEADLESS=1 python train.py --config experiments/hunt_cow/100K_CLIP_HuntCow.yaml
MINEDOJO_HEADLESS=1 python train.py --config experiments/hunt_cow/100K_MLP_HuntCow.yaml
