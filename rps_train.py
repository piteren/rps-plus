from pypaq.lipytools.printout import stamp, ProgBar
import torch

from rps_envy import ACT_SET, ACT_NAMES, reward_func_vec
from rps_agent import get_agents

setups = {
    #'8_0':          {'n_opt_monpol':8, 'n_opt':0},
    #'5_5':          {'n_opt_monpol':5, 'n_opt':5},
    #'1_9':          {'n_opt_monpol':1, 'n_opt':9},
    #'0_8':          {'n_opt_monpol':0, 'n_opt':8},
    '1_1_2fixed':   {'n_opt_monpol':1, 'n_opt':1, 'fixed_policies':((0.4,0.4,0.2),(0.35,0.35,0.3))}
}

if __name__ == "__main__":

    device = 'cpu'
    #device = 'cuda'
    n_act = len(ACT_NAMES)

    for sk in setups:
        setup = setups[sk]
        batch_size = setup.pop('batch_size',256)  # number of games all x all, keep this even
        n_batches = 10000

        agents = get_agents(stamp=f'{sk}_{stamp(letters=None)}', device=device, **setup)
        agent_name = list(agents.keys())
        n_agents = len(agent_name)

        # bootstrap hist_policy - average agent policy (over all agents)
        hist_policy = torch.Tensor([1]*n_act)/n_act
        hist_policy = torch.tile(hist_policy,(n_agents,1))                                      # [agent,prob]

        pb = ProgBar(total=n_batches, show_fract=True, show_speed=True, show_eta=True)
        for batch_ix in range(n_batches):

            agent_policy_log = torch.stack([
                agents[k](hist_policy)['logits'] for k in agent_name])                          # [agent,agent,prob]

            dist = torch.distributions.Categorical(logits=agent_policy_log)

            # save hist_policy for next loop
            hist_policy = torch.mean(dist.probs, dim=1)                                         # [agent,probs]

            action = dist.sample((batch_size,))                                                 # [batch, agent,-> agent]
            action_a = action[:batch_size//2]                                                   # [batch//2, agent_a,-> agent_b]
            action_b = action[batch_size//2:].transpose(-2,-1)                                  # as above
            reward_a, reward_b = reward_func_vec(action_a, action_b)                            # as action_a & b

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

            pb(current=batch_ix, prefix='TR:')