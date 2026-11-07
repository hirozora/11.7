"""Action-type heads on M1's encoders, for the player-centric test (docs/MODEL_PLAN.md, implementation note 22).

The context encoder, the candidate encoder and the who-scorer are those of src/seq_model.py (M1). On top of them:
  context      T0: MLP(context) -> action type
  pooled       T1, T1w: MLP([context, attention pooling of the candidate states, query from the context]) -> action type
  conditional  T2/T3: MLP([context, state_k]) per candidate k, read as P(type | context, actor k). It is trained on the
               true actor, i.e. on the factorised joint likelihood log P(k*) + log P(type* | k*); it is evaluated
               with the true actor (T3, a diagnostic ceiling) and as the mixture sum_k P(k) P(type | k) with the
               who-head's P(k) (T2, a legitimate predictor)
The who-head is always computed; its loss enters training when cfg["who_loss"] is true (always for conditional).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from seq_model import M1


class WhoType(nn.Module):
    def __init__(self, c, meta, head, n_types):
        super().__init__()
        self.m1 = M1(c, meta)
        d, dr = c['hidden'], c['dropout']
        self.head, self.d = head, d
        if head == 'pooled':
            self.query = nn.Linear(d, d)
            self.key = nn.Linear(d, d)
        width = d if head == 'context' else 2 * d
        self.type_head = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Dropout(dr), nn.Linear(d, n_types))

    def forward(self, b):
        """who scores (B, C) with -inf padding, and type logits: (B, T) for context and pooled, (B, C, T) for
        conditional."""
        c = self.m1.encode_context(b)
        state = self.m1.candidate_states(b)
        who = self.m1.who_scores(b, c, state)
        if self.head == 'context':
            return who, self.type_head(c)
        if self.head == 'pooled':
            att = (self.query(c)[:, None, :] * self.key(state)).sum(-1) / math.sqrt(self.d)
            att = att.masked_fill(~b['cand_mask'], float('-inf')).softmax(-1)
            return who, self.type_head(torch.cat([c, (att[..., None] * state).sum(1)], -1))
        C = state.shape[1]
        return who, self.type_head(torch.cat([c[:, None, :].expand(-1, C, -1), state], -1))


def type_log_probs(model, who, logits, actor=None):
    """Log P(type) per target, (B, T). For the conditional head: the mixture over the who-head's P(k), or with actor
    given (true actor slots, all >= 0) the conditional of that actor."""
    if model.head != 'conditional':
        return F.log_softmax(logits, -1)
    cond = F.log_softmax(logits, -1)  # (B, C, T)
    if actor is not None:
        return cond[torch.arange(len(actor), device=cond.device), actor]
    lw = F.log_softmax(who, -1)  # -inf on padding
    return torch.logsumexp(lw[..., None] + cond.masked_fill(torch.isinf(lw)[..., None], float('-inf')), 1)
