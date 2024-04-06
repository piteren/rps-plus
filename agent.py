import random
import torch
from torchness.types import TNS, DTNS
from torchness.motorch import MOTorch, Module
from torchness.layers import LayDense,zeroes
from torchness.tbwr import TBwr
from typing import List, Optional, Tuple

from envy import ACT_SET

# TODO:
#  experiment with hpms

class RPSAgent(Module):

    def __init__(
            self,
            num_classes: int,
            in_lay_norm=            False,
            n_hidden: int=          1,
            hidden_width: int=      30,
            activation=             torch.nn.Tanh,
            lay_norm=               False,
            monpol=                 True,   # monitor (or not) opponent policy (given as inp:TNS)
            baseLR=                 1e-4,
            reward_scale: float=    0.1,
            opt_class=              torch.optim.Adam,
            #opt_alpha=              0.7,
            #opt_beta=               0.5,
            do_zeroes=              False,
            device=                 'cuda',
            **kwargs):

        super().__init__(**kwargs)

        lay_shapeL = []
        next_in = num_classes
        for _ in range(n_hidden):
            lay_shapeL.append((next_in,hidden_width))
            next_in = hidden_width

        self.ln = torch.nn.LayerNorm(num_classes) if in_lay_norm else None

        self.linL = [LayDense(*shape, activation=activation) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) if lay_norm else None for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            if ln: self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.logits = LayDense(
            in_features=    hidden_width,
            out_features=   num_classes,
            activation=     None,
            bias=           False)
        
        self.inp_pad = torch.Tensor([1]*num_classes).to(device) / num_classes
        self.monpol = monpol
        self.reward_scale = reward_scale
        self.do_zeroes = do_zeroes

    def forward(self, inp:TNS) -> DTNS:

        if not self.monpol:
            inp = self.inp_pad.repeat(len(inp), 1)

        out = inp
        if self.ln:
            out = self.ln(out)

        zsL = []
        for lin, ln in zip(self.linL, self.lnL):
            out = lin(out)
            if self.do_zeroes:
                zsL.append(zeroes(out))
            if ln:
                out = ln(out)

        logits = self.logits(out)
        dist = torch.distributions.Categorical(logits=logits)
        return {
            'logits': logits,
            'probs':  dist.probs,
            'zeroes': torch.cat(zsL).detach() if self.do_zeroes else None}

    def loss(self, action:TNS, reward:TNS, inp:TNS) -> DTNS:
        out = self(inp)
        actor_ce = torch.nn.functional.cross_entropy(
            input=      out['logits'],
            target=     action,
            reduction=  'none')
        out.update({'loss': torch.mean(actor_ce * reward * self.reward_scale)})
        return out


class FixedAgent:

    def __init__(
            self,
            name:str,
            probs:List[float],
            save_topdir=    '_models',
            device=         'cuda'):
        self.name = name
        self.save_topdir = save_topdir
        self.device = device
        self.logits = torch.log(torch.Tensor(probs)).to(self.device)
        self.baseLR = 0

        self.train_step = 0
        self._TBwr = TBwr(logdir=f'{self.save_topdir}/{self.name}')

    def __call__(self, inp:Optional[TNS], *args, **kwargs):
        return self.forward(inp)

    def randomize_policy(self):
        probs = torch.rand(3)
        probs = probs / sum(probs)
        self.logits = torch.log(probs).to(self.device)

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


def get_random_policy(n:Optional[int]=None) -> TNS:
    if n: probs = torch.rand(n,3)
    else: probs = torch.rand(3)
    return probs / probs.sum(dim=-1, keepdim=True)


def get_agents(
        stamp: str,
        n_opt_monpol: int=                  0,
        n_opt: int=                         0,
        fixed_policies: Tuple[Tuple,...]=   (),
        lr_range=                           (1e-3, 3e-4),
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
        agents[f'a_{stamp}_f{ix}'] = FixedAgent(name=f'a_{stamp}_f{ix}', probs=policy, device=device)

    return agents