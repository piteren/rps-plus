from torchness.motorch import MOTorch
import unittest

from agent import RPSAgent, get_random_policy

MODELS_DIR = 'tests/_models'


class TestAgent(unittest.TestCase):

    def test_agent_init(self):
        MOTorch(module_type=RPSAgent, num_classes=3, save_topdir=MODELS_DIR, loglevel=10)

    def test_get_random_policy(self):

        policy = get_random_policy()
        print(policy, policy.sum(dim=-1))

        policy = get_random_policy(4)
        print(policy, policy.sum(dim=-1))