from policy import  Policy
import logging
import torch.nn as nn


def mlp(input_dim,mlp_dims, last_relu = False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) -1 ):
        layers.append(nn.Linear(mlp_dims[i],mlp_dims[i+1]))
        
        if i != (len(mlp_dims)-2 ) or last_relu :
            # print('hhh')
            layers.append(nn.ReLU())

    net  = nn.Sequential(*layers)
        

        

class ValueNetwork(nn.Module):
    def __init__(self,input_dim,mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim,mlp_dims)

    def forward(self,state):
        vlaue = self,value_network(state)
        return value



class CADRL(Policy):
    def __init__(self):
        super().__init__()
        self.name  = "CADRL"
        self.trainable = True
        self.multiagent_traning = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space  = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 5
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim+self.human_state_dim
        
    def config(self,config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('cadrl', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim,mlp_dims)
        self.multiagent_traning = config.getboolean('cadrl','multiagent_training')
        logging.info("Cadrl config done")

    def set_common_parameters(self,config):
        #delete some params to further cleairify the result
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling') #?
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self,epsilon):
        self.epsilon = epsilon

    def build_action_space(self,v_pref):
        pass