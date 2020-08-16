import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]

        self.mlp1 = mlp(input_dim + 4, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        # self.lstm1 = nn.LSTM(input_dim, 3,batch_first=True)

        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.mlp3a = mlp(mlp3_input_dim, [150, 100, 80])
        self.attention_weights = None
        self.advantage_stream = nn.Sequential(
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 1)
        )

    def LocalMap(self, state):
        Wholemap = torch.zeros(state.shape[0], state.shape[1], state.shape[2] + 4).cuda()
        CellSize = 3
        sampleNumber = state.shape[0]
        for i in range(sampleNumber):
            SampleX_Y = state[i][:, 6:8]
            SampleN = SampleX_Y.shape[0]
            for j in range(SampleN):
                OtherLocation = torch.stack([x for y, x in enumerate(SampleX_Y) if y != j])
                CurrentLocation = SampleX_Y[j]
                RelativeLocation = OtherLocation - CurrentLocation
                RelativeValue = torch.norm(RelativeLocation, dim=1)
                map = torch.zeros(2, 2).cuda()
                for k in range(RelativeValue.shape[0]):
                    if RelativeValue[k] < CellSize:
                        if RelativeLocation[k][0] > 0:
                            if RelativeLocation[k][1] > 0:
                                map[0][0] += 1
                            else:
                                map[0][1] += 1
                        else:
                            if RelativeLocation[k][1] > 0:
                                map[1][0] += 1
                            else:
                                map[1][1] += 1

                map = map.flatten()
                Wholemap[i][j] = torch.cat([state[i][j], map])

        return Wholemap

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        state = self.LocalMap(state)
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        adv = self.mlp3a(joint_state)
        advantage = self.advantage_stream(adv)

        qvals = value + (advantage - advantage.mean())

        return qvals


class SARL_L(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights


if __name__ == '__main__':
    a = torch.zeros((100, 8, 13))

