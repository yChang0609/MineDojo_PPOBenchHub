CONFIG_FILE=configs/1M_CNN_CombatSpider.yaml
MINEDOJO_HEADLESS=1 python train.py --config $CONFIG_FILE
MINEDOJO_HEADLESS=1 python eval.py --config $CONFIG_FILE