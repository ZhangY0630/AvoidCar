import logging
import sys
import torch
from cadrl import CADRL
import configparser
import gym
import crowd_sim
def main():
    il_weight_path = "./il_model.pth"
    rl_weight_path = './rl_model.pth'
    log_path = './output.log'

    file_handler = logging.FileHandler(log_path,mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level = level, handlers = [file_handler,stdout_handler], format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt = "%Y-%m-%d %H:%M:%S")
    logging.info('test')
    device = torch.device("cpu")
    logging.info(f"Using device : {device}")

    policy = CADRL()
    policy_config = configparser.RawConfigParser()
    policy_config.read('policy.config')
    policy.config(policy_config)
    policy.set_device(device)

    env_config = configparser.RawConfigParser()
    env = gym.make('CrowdSim-v0')


if __name__ == "__main__":
    main()