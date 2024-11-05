CONFIG_FILE=configs/hunt_cow_config.yaml
MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE