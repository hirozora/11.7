"""M1: the identity-free sequence model of the who-acts head (docs/MODEL_PLAN.md, section 3).

Vendored from graph_prediction_research (src/model.py Layer; signal_screening_v1 ContextEncoder readout and scorer),
adapted to the arrays of src/model_inputs.py:
  context   two Pre-LN attention layers (64 wide, 4 heads) over the target's window of 32 records, with a learned bias
            on (position difference, log-dt difference, dt valid); readout = last valid token and masked mean, joined
            with an embedding of the context summary
  candidate MLP over the candidate's categorical and numeric state (including the pooling of his own earlier actions)
  score     MLP([context, candidate]) + w[position group] * U (the confirmed urgency x position-group form)
With c['history_inputs'] false (M1-, docs/MODEL_PLAN.md implementation note 19) the candidate inputs computed from earlier
matches are left out: the prematch role and its position group (cand_cat columns 0 and 2) and the prior matches, prior
events and distance from the prior mean position to the target start with its flag (cand_num columns 6-9); the
provider's static role (GK/DF/MD/FW) and the goalkeeper flag stay, and the urgency term is grouped by the static role.
With c['target_start'] false (the rough start-unknown test, note 23) the context summary leaves out the target's
zone and x/y (ctx_cat column 0, ctx_num columns 0-1) and the possession start x (ctx_num column 3, the target's own x
when the target starts the possession); the target's time stays known.
With c['provider_role'] false as well, the provider's static role (cand_cat column 1, from a players.json snapshot taken
after the season) is left out too and the urgency term is grouped by the goalkeeper flag (implementation note 20).
Scores of padding slots are -inf; the loss is the negative log-softmax of the actor's slot.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Layer(nn.Module):
    """Pre-LN attention block with a learned pairwise bias (graph_prediction_research/src/model.py, unchanged)."""

    def __init__(self, c):
        super().__init__(); d = c['hidden']; self.heads = c['heads']; self.width = d // self.heads
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.out = nn.Linear(d, d)
        self.bias = nn.Sequential(nn.Linear(c['channels'], c['bias_hidden']), nn.GELU(), nn.Linear(c['bias_hidden'], self.heads))
        self.ln2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, c['ffn']), nn.GELU(), nn.Dropout(c['dropout']), nn.Linear(c['ffn'], d))
        self.drop = nn.Dropout(c['dropout'])

    def forward(self, x, channels, allowed, valid):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, n, 3, self.heads, self.width).permute(2, 0, 3, 1, 4).unbind(0)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(self.width) + self.bias(channels).permute(0, 3, 1, 2)
        attention = score.masked_fill(~allowed[:, None], float('-inf')).softmax(-1)
        message = (self.drop(attention) @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.out(message)); x = x + self.drop(self.ff(self.ln2(x)))
        return x * valid[:, :, None]


def projection(i, o, dropout):
    return nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(dropout))


HISTORY_NUM = [6, 7, 8, 9]  # cand_num columns from earlier matches (src/model_inputs.py)


class M1(nn.Module):
    def __init__(self, c, meta):
        super().__init__()
        self.history, self.provider = c['history_inputs'], c['provider_role']
        d, dr = c['hidden'], c['dropout']
        voc = meta['vocab']
        self.ev = nn.Embedding(len(voc['event_name']) + 1, 16, padding_idx=0)
        self.sub = nn.Embedding(len(meta['sub_event_names']) + 1, 16, padding_idx=0)
        self.side = nn.Embedding(3, 4, padding_idx=0)
        self.zone = nn.Embedding(10, 8)
        self.token = projection(16 + 16 + 4 + 8 + 6, d, dr)
        self.layers = nn.ModuleList([Layer({**c, 'channels': 3}) for _ in range(c['layers'])])
        self.norm = nn.LayerNorm(d)
        self.readout = projection(2 * d, d, dr)
        sizes = [9, len(voc['phase']), len(voc['prev_event']), len(voc['opp_last_event']), len(voc['possession_start']),
                 len(voc['venue']), len(voc['opponent_class'])]
        self.start = c['target_start']
        self.ctx_cat_cols = list(range(len(sizes))) if self.start else list(range(1, len(sizes)))
        num_keep = list(range(7)) if self.start else [2, 4, 5, 6]
        self.register_buffer('ctx_num_keep', torch.tensor(num_keep), persistent=False)
        self.ctx_emb = nn.ModuleList([nn.Embedding(sizes[j], 8) for j in self.ctx_cat_cols])
        self.summary = projection(8 * len(self.ctx_cat_cols) + len(num_keep), d, dr)
        self.context = projection(2 * d, d, dr)
        # modules are created in the same order as before the history switch, so that M1 itself is unchanged
        if self.history:
            self.role = nn.Embedding(len(meta['roles']) + 1, 8)
        if self.provider:
            self.wy = nn.Embedding(4, 4)
        if self.history:
            self.group = nn.Embedding(len(meta['groups']), 4)
        self.gk = nn.Embedding(2, 2)
        keep = [j for j in range(meta['cand_num']) if self.history or j not in HISTORY_NUM]
        self.register_buffer('num_keep', torch.tensor(keep), persistent=False)
        width = (8 + 4 if self.history else 0) + (4 if self.provider else 0) + 2 + len(keep)
        self.cand = nn.Sequential(projection(width, d, dr), projection(d, d, dr))
        self.scorer = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dr), nn.Linear(d, 1))
        self.urgency = nn.Parameter(torch.zeros(len(meta['groups']) if self.history else (4 if self.provider else 2)))

    def encode_context(self, b):
        valid = b['win_valid']
        cat = b['win_cat'].long()
        x = self.token(torch.cat([self.ev(cat[..., 0]), self.sub(cat[..., 1]), self.side(cat[..., 2]), self.zone(cat[..., 3]),
                                  b['win_num']], -1))
        n = valid.shape[1]
        pos, ldt, dt_ok = b['win_num'][..., 5], b['win_num'][..., 3], b['win_num'][..., 4] > 0
        pair_ok = valid[:, :, None] & valid[:, None, :]
        channels = torch.stack([pos[:, :, None] - pos[:, None, :],
                                torch.where(dt_ok[:, :, None] & dt_ok[:, None, :], ldt[:, :, None] - ldt[:, None, :], 0.0),
                                (dt_ok[:, :, None] & dt_ok[:, None, :] & pair_ok).float()], -1)
        eye = torch.eye(n, dtype=torch.bool, device=valid.device)[None]
        allowed = pair_ok | (~valid[:, :, None] & eye)  # padded queries attend to themselves only
        for layer in self.layers:
            x = layer(x, channels, allowed, valid)
        h = self.norm(x)
        last = h[:, -1]  # windows are left-padded, so the last slot is the most recent record when any exists
        mean = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        r = self.readout(torch.cat([last, mean], -1))
        cc = b['ctx_cat'].long()
        s = self.summary(torch.cat([e(cc[:, j]) for j, e in zip(self.ctx_cat_cols, self.ctx_emb)] + [b['ctx_num'][:, self.ctx_num_keep]], -1))
        return self.context(torch.cat([r, s], -1))

    def candidate_states(self, b):
        """(B, C, hidden) representation of every candidate slot (padding included; masked when scored)."""
        k = b['cand_cat'].long()
        parts = ([self.role(k[..., 0])] if self.history else []) + ([self.wy(k[..., 1])] if self.provider else [])
        parts += ([self.group(k[..., 2])] if self.history else []) + [self.gk(k[..., 3])]
        parts.append(b['cand_num'] if self.history else b['cand_num'][..., self.num_keep])
        return self.cand(torch.cat(parts, -1))

    def who_scores(self, b, c, state):
        k = b['cand_cat'].long()
        C = state.shape[1]
        scores = self.scorer(torch.cat([c[:, None, :].expand(-1, C, -1), state], -1)).squeeze(-1)
        group = k[..., 2] if self.history else (k[..., 1] if self.provider else k[..., 3])
        scores = scores + self.urgency[group] * b['urgency'][:, None]
        return scores.masked_fill(~b['cand_mask'], float('-inf'))

    def forward(self, b):
        c = self.encode_context(b)
        return self.who_scores(b, c, self.candidate_states(b)), c


def actor_loss(scores, actor_slot):
    """Negative log-softmax of the actor's slot; every row must be reachable."""
    return -F.log_softmax(scores, -1).gather(1, actor_slot[:, None]).squeeze(1)
