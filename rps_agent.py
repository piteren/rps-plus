import random
import torch
from torchness.types import TNS, DTNS
from torchness.motorch import MOTorch
from torchness.models.simple_feats_classifier import SFeatsCSF
from torchness.tbwr import TBwr
from typing import List, Optional, Tuple

from rps_envy import ACT_SET


# TODO:
#  experiment with hpms
#  add zeroes
#  add inp / out hist



class RPSAgent(SFeatsCSF):

    def __init__(
            self,
            num_classes: int,
            in_lay_norm=            False,
            n_hidden: int=          1,
            hidden_width: int=      10,
            lay_norm=               True,
            monpol=                 True,   # monitor (or not) opponent policy (given as inp:TNS)
            baseLR=                 1e-4,
            reward_scale: float=    0.1,
            opt_class=              torch.optim.Adam,
            opt_alpha=              0.7,
            opt_beta=               0.5,
            device=                 'cuda',
            **kwargs):

        super().__init__(
            feats_width=    num_classes,
            in_lay_norm=    in_lay_norm,
            n_hidden=       n_hidden,
            hidden_width=   hidden_width,
            num_classes=    num_classes,
            lay_norm=       lay_norm,
            **kwargs)
        self.inp_pad = torch.Tensor([1]*num_classes).to(device) / num_classes
        self.monpol = monpol
        self.reward_scale = reward_scale

    def forward(self, inp:Optional[TNS]=None) -> DTNS:

        if inp is None:
            inp = self.inp_pad
        else:
            if not self.monpol:
                inp = self.inp_pad.repeat(len(inp), 1)

        fwd_out = super().forward(feats=inp)
        dist = torch.distributions.Categorical(logits=fwd_out['logits'] )
        fwd_out.update({'probs':dist.probs})
        return fwd_out

    def loss(self, action, reward, inp:Optional[TNS]=None) -> DTNS:

        if inp is None:
            inp = self.inp_pad.repeat(len(action), 1)

        out = self(inp)
        actor_ce = torch.nn.functional.cross_entropy(
            input=      out['logits'],
            target=     action,
            reduction=  'none')
        out.update({'loss': torch.mean(actor_ce * reward * self.reward_scale)})
        return out


class FAgent:

    def __init__(
            self,
            name:str,
            probs:List[float],
            save_topdir=    '_models',
            device=         'cuda'):
        self.name = name
        self.save_topdir = save_topdir
        self.logits = torch.log(torch.Tensor(probs)).to(device)
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
        lr_range=                           (1e-4, 3e-5),
        rand=                               True,
        device=                             'cuda',
):

    agents = {f'a_{stamp}_{x:02}': MOTorch(
        module_type=    RPSAgent,
        name=           f'a_{stamp}_{x:02}',
        num_classes=    len(ACT_SET),
        monpol=         x < n_opt_monpol,
        seed=           random.randint(0, 100000) if rand else 123,
        baseLR=         (lr_range[0]+random.random()*(lr_range[1]-lr_range[0])),
        device=         device,
    ) for x in range(n_opt_monpol + n_opt)}

    for ix,policy in enumerate(fixed_policies):
        agents[f'a_{stamp}_f{ix}'] = FAgent(name=f'a_{stamp}_f{ix}', probs=policy, device=device)

    return agents


if __name__ == "__main__":

    agent = MOTorch(module_type=RPSAgent, num_classes=3, loglevel=10)
    print(agent)