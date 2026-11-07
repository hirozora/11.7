"""Native-input encoder ports with conditional actor heads; see docs/ACTOR_BASELINES.md.

Only the output head sees query time/start and candidate identities. No prematch
history, roles, scenario summaries or candidate action pools enter these models.
"""
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ('seq2event', 'nmstpp', 'lem')


def positional_encoding(length, width, device, dtype):
    pos = torch.arange(length, dtype=torch.float64, device=device)[:, None]
    col = torch.arange(width, device=device)
    angle = pos / 100.0 ** (2 * col.double() / width)
    return torch.where(col % 2 == 0, torch.sin(angle), torch.cos(angle)).to(dtype)


def attention_inputs(x, valid):
    safe = valid.clone()
    empty = ~valid.any(1)
    safe[empty, -1] = True
    return x.masked_fill(~valid[..., None], 0.0), safe, empty


class Seq2EventHistory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(7, cfg['action_embedding'], padding_idx=1, scale_grad_by_freq=True)
        self.lin0 = nn.Linear(10, cfg['continuous_embedding'])
        width = cfg['action_embedding'] + cfg['continuous_embedding']
        layer = nn.TransformerEncoderLayer(width, cfg['heads'], cfg['ffn'], cfg['dropout'], batch_first=True)
        self.seqnet = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        self.lin1 = nn.Linear(width, cfg['readout'])
        self.out_width = cfg['readout']

    def forward(self, b):
        v = b['valid']
        cat = b['actions'].long().masked_fill(~v, 0)
        num = b['continuous'].masked_fill(~v[..., None], 0.0)
        x = torch.cat([self.emb(cat), self.lin0(num)], -1)
        x, safe, empty = attention_inputs(x, v)
        x = x + positional_encoding(x.shape[1], x.shape[2], x.device, x.dtype)[None]
        h = self.seqnet(x, src_key_padding_mask=~safe)[:, -1]
        return F.relu(self.lin1(h)).masked_fill(empty[:, None], 0.0)


class NMSTPPHistory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_act = nn.Embedding(5, 5, scale_grad_by_freq=True)
        self.emb_zone = nn.Embedding(20, 20, scale_grad_by_freq=True)
        self.lin0 = nn.Linear(6, 6)
        self.encoder_layer = nn.TransformerEncoderLayer(31, cfg['heads'], cfg['ffn'], cfg['dropout'], batch_first=True)
        self.lin_relu = nn.Linear(31, 31)  # Upstream applies no ReLU at this layer.
        self.out_width = 31

    def forward(self, b):
        v = b['valid']
        x = torch.cat([self.emb_act(b['actions'].long().masked_fill(~v, 0)),
                       self.emb_zone(b['zones'].long().masked_fill(~v, 0)),
                       self.lin0(b['continuous'].masked_fill(~v[..., None], 0.0))], -1)
        x, safe, empty = attention_inputs(x, v)
        # Sequence positions, correcting the upstream batch-axis encoding defect.
        x = x + positional_encoding(x.shape[1], x.shape[2], x.device, x.dtype)[None]
        h = self.encoder_layer(x, src_key_padding_mask=~safe)[:, -1]
        return self.lin_relu(h).masked_fill(empty[:, None], 0.0)


class LEMHistory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Top-level Unified-LEM tabular MLP, not its older BatchNorm/sigmoid variant.
        widths = [15 + 6 * cfg['window'], *cfg['hidden_sizes']]
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(cfg['dropout'])]
        self.model = nn.Sequential(*layers)
        self.out_width = widths[-1]

    def forward(self, b):
        return self.model(b['tabular'])


class ActorBaseline(nn.Module):
    def __init__(self, family, cfg, players):
        super().__init__()
        encoders = dict(seq2event=Seq2EventHistory, nmstpp=NMSTPPHistory, lem=LEMHistory)
        self.history = encoders[family](cfg['encoders'][family])
        self.actor = nn.Linear(self.history.out_width + 4, players)

    def encode_history(self, b):
        """Native representation reusable by separately defined event output heads."""
        return self.history(b)

    def forward(self, b):
        h = self.encode_history(b)
        known = self.actor(torch.cat([h, b['query']], -1))
        # Explicit open-vocabulary rule: every unseen candidate gets the SAME mean
        # trained-class logit. No randomly initialised unseen output weights.
        logits = torch.cat([known.mean(1, keepdim=True), known], 1)
        scores = logits.gather(1, b['cand_id'].long())
        return scores.masked_fill(~b['cand_mask'], -torch.inf), h
