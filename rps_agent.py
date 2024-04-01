import random
import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module, MOTorch
from torchness.layers import LayDense
from torchness.tbwr import TBwr
from typing import List, Optional

from rps_envy import ACT_SET


# TODO:
#  why device does not work?
#  add zeroes
#  add inp / out hist



class RPSAgent(Module):

    def __init__(
            self,
            n_actions:int,
            monpol=     True,
            baseLR=     1e-3,
            device=     None,
            **kwargs):
        super().__init__(**kwargs)
        self.inp_pad = torch.Tensor([1]*n_actions) / n_actions
        self.inp_pad = self.inp_pad.to('cuda:1') # TODO

        self.lay = LayDense(n_actions,20)
        self.logits = LayDense(20,n_actions,activation=None)

        self.monpol = monpol
        self.policy_mavg = torch.Tensor([1]*n_actions) / n_actions
        self.policy_mavg = self.policy_mavg.to('cuda:1')  # TODO

    def forward(self, inp:Optional[TNS]=None, force_inp=False) -> DTNS:

        if not self.monpol and not force_inp:
            inp = None
        if inp is None:
            inp = self.inp_pad

        md = self.lay(inp)
        lg = self.logits(md)

        dist = torch.distributions.Categorical(logits=lg)
        probs = dist.probs

        # update policy_mavg only when running test (not train)
        if len(probs.shape) == 1:
            self.policy_mavg = 0.99 * self.policy_mavg + 0.01 * probs

        return {'logits': lg, 'probs': probs}

    def loss(self, action, reward, inp:Optional[TNS]=None) -> DTNS:

        if not self.monpol:
            inp = None
        if inp is None:
            inp = self.inp_pad
            inp = inp.repeat(len(action), 1)

        out = self(inp, force_inp=True)
        actor_ce = torch.nn.functional.cross_entropy(
            input=      out['logits'],
            target=     action,
            reduction=  'none')
        out.update({'loss': torch.mean(actor_ce * reward)})
        return out


class FAgent:

    def __init__(self, name:str, probs:List[float], save_topdir='_models'):
        self.name = name
        self.save_topdir = save_topdir
        self.probs = torch.Tensor(probs)
        self.baseLR = 0

        self.train_step = 0
        self._TBwr = TBwr(logdir=f'{self.save_topdir}/{self.name}')

    def __call__(self, *args, **kwargs):
        return self.forward()

    def forward(self):
        return {'probs': self.probs}

    def backward(self, *args, **kwargs):
        self.train_step += 1
        return {'loss':0, 'gg_norm':0}

    @property
    def policy_mavg(self):
        return self.probs

    def log_TB(self, value, tag:str, step:Optional[int]=None):
        """ logs value to TB """
        if step is None:
            step = self.train_step
        self._TBwr.add(value=value, tag=tag, step=step)


class RPSA_MOTorch(MOTorch):
    @property
    def policy_mavg(self):
        return self.module.policy_mavg


def get_agents(setup:str, st:str):

    if setup == 'opt_bad_gto':

        n_opt_agents = 2
        lrr = [5e-4, 5e-5]
        agents = {f'a_{st}_{x:02}': RPSA_MOTorch(
            module_type=    RPSAgent,
            name=           f'a_{st}_{x:02}',
            n_actions=      len(ACT_SET),
            monpol=         bool(x % 2),
            #baseLR=         (lrr[0]+random.random()*(lrr[1]-lrr[0])),
        ) for x in range(n_opt_agents)}

        agents[f'a_{st}_fa'] = FAgent(name=f'a_{st}_fa', probs=[0.4, 0.4, 0.2])
        agents[f'a_{st}_fb'] = FAgent(name=f'a_{st}_fb', probs=[0.35, 0.35, 0.3])
        return agents

    if setup == 'monpol_and_not':

        n_agents = 10
        lrr = [1e-3, 5e-5]
        agents = {f'a_{st}_{x:02}': RPSA_MOTorch(
            module_type=    RPSAgent,
            name=           f'a_{st}_{x:02}',
            n_actions=      len(ACT_SET),
            monpol=         x < 5,
            baseLR=         (lrr[0]+random.random()*(lrr[1]-lrr[0])),
        ) for x in range(n_agents)}
        return agents