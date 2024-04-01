ACT_NAMES = ['0_ROCK','1_PAPER','2_SCISSORS']
ACT_SET = list(range(len(ACT_NAMES)))


def reward_func(action_a:int, action_b:int, plus=2) -> tuple:
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