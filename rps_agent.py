import random
import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module, MOTorch
from torchness.layers import LayDense
from torchness.tbwr import TBwr
from typing import List, Optional, Tuple

from rps_envy import ACT_SET


# TODO:
#  why device does not work?
#  add zeroes
#  add inp / out hist



class RPSAgent(Module):

    def __init__(
            self,
            n_actions:int,
            monpol=     True,   # monitor (or not) opponent policy (given as inp:TNS)
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
        self.logits = torch.log(torch.Tensor(probs)).to('cuda:1')
        self.baseLR = 0

        self.train_step = 0
        self._TBwr = TBwr(logdir=f'{self.save_topdir}/{self.name}')

    def __call__(self, inp:Optional[TNS], *args, **kwargs):
        return self.forward(inp)

    def forward(self, inp:Optional[TNS]):
        logits = self.logits
        if inp is not None:
            logits = logits.repeat(len(inp), 1)
        return {'logits': logits}

    def backward(self, *args, **kwargs):
        self.train_step += 1
        return {'loss':0, 'gg_norm':0}

    def log_TB(self, value, tag:str, step:Optional[int]=None):
        """ logs value to TB """
        if step is None:
            step = self.train_step
        self._TBwr.add(value=value, tag=tag, step=step)


def get_agents(
        stamp: str,
        n_opt_monpol: int,
        n_opt: int,
        fixed_policies: Tuple[Tuple,...]=   (),
        lr_range=                           (3e-3, 1e-4),
        rand=                               True):

    agents = {f'a_{stamp}_{x:02}': MOTorch(
        module_type=    RPSAgent,
        name=           f'a_{stamp}_{x:02}',
        n_actions=      len(ACT_SET),
        monpol=         x < n_opt_monpol,
        seed=           random.randint(0, 100000) if rand else 123,
        baseLR=         (lr_range[0]+random.random()*(lr_range[1]-lr_range[0])),
    ) for x in range(n_opt_monpol + n_opt)}

    for ix,policy in enumerate(fixed_policies):
        agents[f'a_{stamp}_f{ix}'] = FAgent(name=f'a_{stamp}_f{ix}', probs=policy)

    return agents