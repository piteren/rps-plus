from pypaq.lipytools.printout import stamp, ProgBar
import torch

from envy import ACT_SET, ACT_NAMES, reward_func_vec
from agent import get_agents, FixedAgent, get_random_policy

SETUPS = {
    '0_8':          {                  'n_opt':8},
    '0_8_noplus':   {                  'n_opt':8, 'plus':1},
    '1_7':          {'n_opt_monpol':1, 'n_opt':7},
    '4_4':          {'n_opt_monpol':4, 'n_opt':4},
    '4_4_noplus':   {'n_opt_monpol':4, 'n_opt':4, 'plus':1},
    '8_0':          {'n_opt_monpol':8},
    '1_1_2fix': { # ...against two fixed ones, where one has optimal policy against unknown player
        'n_opt_monpol':     1,
        'n_opt':            1,
        'fixed_policies':   ((0.4,0.4,0.2), (0.35,0.35,0.3)),
    },
    '1_1_2fix_noplus': { # ...against two fixed ones, without plus
        'n_opt_monpol':     1,
        'n_opt':            1,
        'fixed_policies':   ((0.5,0.3,0.2), (0.2,0.7,0.1)),
        'plus':             1,
    },
    '1_7random': { # 1 monpol against 9 fixed randomized
        'n_opt_monpol':     1,
        'fixed_policies':   tuple([(1/3,1/3,1/3)]*7),
        'randomize_fixed':  True,
    },
    '1_7random_noplus': { # 1 monpol against 9 fixed randomized, without plus
        'n_opt_monpol':     1,
        'fixed_policies':   tuple([(1/3,1/3,1/3)]*7),
        'randomize_fixed':  True,
        'plus':             1,
    },
}


if __name__ == "__main__":

    device = 'cpu'
    #device = 'cuda'
    n_act = len(ACT_NAMES)

    for sk in [
        '0_8',
        '0_8_noplus',
        '1_7',
        '4_4',
        '4_4_noplus',
        '8_0',
        '1_1_2fix',
        '1_1_2fix_noplus',
        '1_7random',
        '1_7random_noplus',
    ]:

        d = {
            'batch_size':       256,
            'n_batches':        5000,
            'randomize_fixed':  False,
            'plus':             2}

        setup = SETUPS[sk]

        for k in d:
            if k in setup:
                d[k] = setup.pop(k)

        agents = get_agents(stamp=f'{sk}_{stamp(letters=None)}', device=device, **setup)
        agent_name = list(agents.keys())
        n_agents = len(agent_name)

        policy_obs = None # agent policy (estimation) against all agents, using some state observation

        pb = ProgBar(total=d['n_batches'], show_fract=True, show_speed=True, show_eta=True)
        for batch_ix in range(d['n_batches']):

            if d['randomize_fixed']:
                for k in agent_name:
                    if type(agents[k]) is FixedAgent:
                        agents[k].randomize_policy()

            # try to estimate/update agent policy (with 10 random states given)
            if policy_obs is None or d['randomize_fixed']:
                agent_policy_log = torch.stack([
                    agents[k](inp=get_random_policy(10))['logits'] for k in agent_name])        # [agent,r10,logits]
                dist = torch.distributions.Categorical(logits=agent_policy_log)
                policy_obs = torch.mean(dist.probs, dim=1)                                      # [agent,probs]

            # get agent policy_log against each other policy_obs
            agent_policy_log = torch.stack([
                agents[k](inp=policy_obs)['logits'] for k in agent_name])                       # [agent,agent,logits]

            dist = torch.distributions.Categorical(logits=agent_policy_log)

            # update policy observation
            probs = dist.probs                                                                  # [agent,agent,probs]
            policy_obs = torch.mean(probs, dim=1)                                               # [agent,probs]
            policy_obs_std = torch.std(probs, dim=1)                                            # [agent,probs_std]

            action = dist.sample((d['batch_size'],))                                            # [batch, agent,-> agent]
            action_a = action[:d['batch_size'] // 2]                                            # [batch//2, agent_a,-> agent_b]
            action_b = action[d['batch_size'] // 2:].transpose(-2, -1)                          # as above
            reward_a, reward_b = reward_func_vec(action_a, action_b, plus=d['plus'])            # as action_a & b

            action = torch.concatenate([action_a, action_b.transpose(-2,-1)], dim=0)            # [batch, agent,-> agent]
            reward = torch.concatenate([reward_a, reward_b.transpose(-2,-1)], dim=0)            # [batch, agent,-> agent]
            action = torch.squeeze(torch.concatenate(torch.split(action, 1, dim=0), dim=-1))    # [agent,-> agent*batch]
            reward = torch.squeeze(torch.concatenate(torch.split(reward, 1, dim=0), dim=-1))    # [agent,-> agent*batch]

            policy_obs_repeat = policy_obs.repeat(d['batch_size'], 1)                           # [agent*batch,prob]
            reward_mean = reward.float().mean(dim=-1)
            for ix,k in enumerate(agent_name):

                out = agents[k].backward(action=action[ix], reward=reward[ix], inp=policy_obs_repeat)

                agents[k].log_TB(out['loss'],                 tag='opt/loss')
                agents[k].log_TB(out['gg_norm'],              tag='opt/gg_norm')

                agents[k].log_TB(agents[k].baseLR,            tag='opt/baseLR')
                agents[k].log_TB(reward_mean[ix],             tag='policy/reward')
                for aix,anm in zip(ACT_SET,ACT_NAMES):
                    agents[k].log_TB(policy_obs[ix][aix],     tag=f'policy/p{aix}_{anm}')
                    agents[k].log_TB(policy_obs_std[ix][aix], tag=f'policy/pstd{aix}_{anm}')

            pb(current=batch_ix+1, prefix='TR:')