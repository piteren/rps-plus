from typing import Tuple
import torch
from torchness.types import TNS

PLUS = 2
ACT_NAMES = ['0_ROCK','1_PAPER','2_SCISSORS']
ACT_SET = list(range(len(ACT_NAMES)))


def reward_func(action_a:int, action_b:int, plus:int=PLUS) -> Tuple[int,int]:
    reward_a = 0
    if action_a == 0:
        if action_b == 1: reward_a = -1
        if action_b == 2: reward_a = +1 * plus
    if action_a == 1:
        if action_b == 0: reward_a = +1
        if action_b == 2: reward_a = -1 * plus
    if action_a == 2:
        if action_b == 0: reward_a = -1 * plus
        if action_b == 1: reward_a = +1 * plus
    return reward_a, -reward_a


def reward_func_vec(action_a:TNS, action_b:TNS, plus:int=PLUS) -> Tuple[TNS,TNS]:
    """ vectorised version of reward function """
    reward_a = torch.where(
        condition=  (torch.gt(action_a,action_b) & ~((action_a == 2) & (action_b == 0))) | (action_a == 0) & (action_b == 2),
        input=      torch.ones_like(action_a),
        other=     -torch.ones_like(action_a))
    reward_a = torch.where(
        condition=  action_a == action_b,
        input=      torch.zeros_like(action_a),
        other=      reward_a)
    if plus > 1:
        reward_a = torch.where(
            condition=  (action_a == 2) | (action_b == 2),
            input=      reward_a * plus,
            other=      reward_a)
    return reward_a, -reward_a


if __name__ == "__main__":

    aa = torch.randint(0, 3, (100,))
    ab = torch.randint(0, 3, (100,))
    for a,b in zip(aa,ab):
        ra, rb = reward_func(a,b)
        ra = int(ra)
        rb = int(rb)
        ra_, rb_ = reward_func_vec(a,b)
        ra_ = int(ra_)
        rb_ = int(rb_)
        print(int(a),int(b), (ra,rb), (ra_,rb_))
        assert (ra,rb) == (ra_,rb_)