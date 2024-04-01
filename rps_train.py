from pypaq.lipytools.printout import stamp, progress_
import torch

from rps_envy import ACT_SET, ACT_NAMES, reward_func_vec
from rps_agent import get_agents


if __name__ == "__main__":

    n_act = len(ACT_NAMES)

    agents = get_agents(
        #'opt_bad_gto',
        #'monpol_and_not',
        'monpol',
        st = stamp(letters=None),
    )
    agent_name = list(agents.keys())
    n_agents = len(agent_name)

    # bootstrap hist_policy - average agent policy (over all agents)
    hist_policy = torch.Tensor([1]*n_act)/n_act
    hist_policy = torch.tile(hist_policy,(n_agents,1))                                      # [agent,prob]

    batch_size = 64 # number of games all x all, keep this even
    n_batches = 50000
    for batch_ix in range(n_batches):

        agent_policy_log = torch.stack([
            agents[k](hist_policy)['logits'] for k in agent_name])                          # [agent,agent,prob]

        dist = torch.distributions.Categorical(logits=agent_policy_log)

        # save hist_policy for next loop
        hist_policy = torch.mean(dist.probs, dim=1)                                         # [agent,probs]

        action = dist.sample((batch_size,))                                                 # [batch, agent,-> agent]
        action_a, action_b = action[:batch_size//2], action[batch_size//2:].transpose(-2,-1)# [batch//2, agent_a,-> agent_b] [batch//2, agent_b,-> agent_a]

        reward_a, reward_b = reward_func_vec(action_a, action_b)                            # [batch//2, agent_a,-> agent_b] [batch//2, agent_b,-> agent_a]

        action = torch.concatenate([action_a, action_b.transpose(-2,-1)], dim=0)            # [batch, agent,-> agent]
        reward = torch.concatenate([reward_a, reward_b.transpose(-2,-1)], dim=0)            # [batch, agent,-> agent]
        action = torch.squeeze(torch.concatenate(torch.split(action, 1, dim=0), dim=-1))    # [agent,-> agent*batch]
        reward = torch.squeeze(torch.concatenate(torch.split(reward, 1, dim=0), dim=-1))    # [agent,-> agent*batch]

        hist_policy_repeat = hist_policy.repeat(batch_size, 1)                              # [agent*batch,prob]
        reward_mean = reward.float().mean(dim=-1)
        for ix,k in enumerate(agent_name):

            out = agents[k].backward(action=action[ix], reward=reward[ix], inp=hist_policy_repeat)

            agents[k].log_TB(out['loss'],               tag='opt/loss')
            agents[k].log_TB(out['gg_norm'],            tag='opt/gg_norm')

            agents[k].log_TB(agents[k].baseLR,          tag='opt/baseLR')
            agents[k].log_TB(reward_mean[ix],           tag='policy/reward')
            for aix,anm in zip(ACT_SET,ACT_NAMES):
                agents[k].log_TB(hist_policy[ix][aix],  tag=f'policy/{anm}')

        progress_(current=batch_ix, total=n_batches, prefix='TR:', show_fract=True)