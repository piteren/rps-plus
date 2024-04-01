import numpy as np
from pypaq.lipytools.printout import stamp, progress_
import random
import torch

from rps_envy import ACT_SET, ACT_NAMES, reward_func
from rps_agent import get_agents



if __name__ == "__main__":

    agents = get_agents(
        #'opt_bad_gto',
        'monpol_and_not',
        st = stamp(letters=None),
    )
    agents_names = list(agents.keys())

    n_batches = 5000
    bs = 64
    bix = 0
    while bix < n_batches:

        actions = {k:[] for k in agents_names}
        rewards = {k:[] for k in agents_names}
        mpols =   {k:[] for k in agents_names}

        while min([len(actions[k]) for k in agents_names]) < bs:

            if len(agents_names) > 1:
                key_sel = random.sample(agents_names, 2)
            else:
                key_sel = agents_names * 2

            inp_sel = [agents[k].policy_mavg for k in reversed(key_sel)]
            for k,i in zip(key_sel,inp_sel):
                mpols[k].append(i)

            pol_sel = [agents[k](inp=i)['probs'].detach().cpu().numpy()
                       for k,i in zip(key_sel,inp_sel)]

            act_sel = [np.random.choice(ACT_SET, p=p) for p in pol_sel]
            for k,a in zip(key_sel,act_sel):
                actions[k].append(a)

            rew_sel = reward_func(*act_sel)
            for k,r in zip(key_sel,rew_sel):
                rewards[k].append(r)

        for k in mpols:
            mpols[k] = torch.stack(mpols[k])

        for k in actions:
            actions[k] = torch.Tensor(actions[k]).long()

        for k in rewards:
            rewards[k] = torch.Tensor(rewards[k]).long()

        for k in agents_names:

            out = agents[k].backward(action=actions[k], reward=rewards[k], inp=mpols[k])

            agents[k].log_TB(out['loss'],               tag='opt/loss')
            agents[k].log_TB(out['gg_norm'],            tag='opt/gg_norm')

            agents[k].log_TB(agents[k].baseLR,          tag='opt/baseLR')
            agents[k].log_TB(rewards[k].float().mean(), tag='policy/reward')
            for aix,anm in zip(ACT_SET,ACT_NAMES):
                agents[k].log_TB(agents[k].policy_mavg[aix], tag=f'policy/{anm}_mavg')

        bix += 1
        progress_(current=bix, total=n_batches, prefix='TR:', show_fract=True)
