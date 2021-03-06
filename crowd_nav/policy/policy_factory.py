from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl_d import SARL_D
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.sarl_l import SARL_L
from crowd_nav.policy.sarl_eye import SARL_E

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['sarl_d'] = SARL_D
policy_factory['sarll'] = SARL_L
policy_factory['sarl_e'] = SARL_E

