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
            monpol=     True,   # monitor (or not) opponent policy (given as inp)
            baseLR=     1e-3,
            device=     None,
            opt_class=  torch.optim.Adam,
            opt_alpha=  0.7,
            opt_beta=   0.5,
            **kwargs):
        super().__init__(**kwargs)
        self.inp_pad = torch.Tensor([1]*n_actions) / n_actions
        self.inp_pad = self.inp_pad.to('cuda:1') # TODO

        self.lay = LayDense(n_actions,20)
        self.logits = LayDense(20,n_actions,activation=None)

        self.monpol = monpol

    def forward(self, inp:Optional[TNS]=None) -> DTNS:

        if inp is None:
            inp = self.inp_pad
        else:
            if not self.monpol:
                inp = self.inp_pad.repeat(len(inp), 1)

        md = self.lay(inp)
        lg = self.logits(md)

        dist = torch.distributions.Categorical(logits=lg)
        probs = dist.probs

        return {'logits': lg, 'probs': probs}

    def loss(self, action, reward, inp:Optional[TNS]=None) -> DTNS:

        if inp is None:
            inp = self.inp_pad.repeat(len(action), 1)

        out = self(inp)
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

    def log_TB(self, value, tag:str, step:Optional[int]=None):
        """ logs value to TB """
        if step is None:
            step = self.train_step
        self._TBwr.add(value=value, tag=tag, step=step)


def get_agents(setup:str, st:str, rand=True):

    seed = 123

    if setup == 'opt_bad_gto':

        n_opt_agents = 2
        lrr = [5e-4, 5e-5]
        agents = {f'a_{st}_{x:02}': MOTorch(
            module_type=    RPSAgent,
            name=           f'a_{st}_{x:02}',
            n_actions=      len(ACT_SET),
            monpol=         bool(x % 2),
            seed=           random.randint(0, 100000) if rand else seed,
            #baseLR=         (lrr[0]+random.random()*(lrr[1]-lrr[0])),
        ) for x in range(n_opt_agents)}

        agents[f'a_{st}_fa'] = FAgent(name=f'a_{st}_fa', probs=[0.4, 0.4, 0.2])
        agents[f'a_{st}_fb'] = FAgent(name=f'a_{st}_fb', probs=[0.35, 0.35, 0.3])
        return agents

    if setup == 'monpol_and_not':

        n_agents = 10
        lrr = [1e-3, 5e-5]
        agents = {f'a_{st}_{x:02}': MOTorch(
            module_type=    RPSAgent,
            name=           f'a_{st}_{x:02}',
            n_actions=      len(ACT_SET),
            monpol=         x < 5,
            seed=           random.randint(0, 100000) if rand else seed,
            baseLR=         (lrr[0]+random.random()*(lrr[1]-lrr[0])),
        ) for x in range(n_agents)}
        return agents

    if setup == 'monpol':

        n_agents = 10
        lrr = [1e-3, 5e-5]
        agents = {f'a_{st}_{x:02}': MOTorch(
            module_type=    RPSAgent,
            name=           f'a_{st}_{x:02}',
            n_actions=      len(ACT_SET),
            monpol=         True,
            seed=           random.randint(0, 100000) if rand else seed,
            baseLR=         (lrr[0]+random.random()*(lrr[1]-lrr[0])),
        ) for x in range(n_agents)}
        return agents