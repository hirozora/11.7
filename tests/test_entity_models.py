import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from entity_models import CELLS, METADATA_MP, NODE_TYPES, EntityModel, FastHGTConv


def random_graph(seed=0, width=16):
    """Random node features for every node type and random edges for every relation of METADATA_MP."""
    g = torch.Generator().manual_seed(seed)
    sizes = {"player": 9, "slot": 6, "team": 3, "role": 4, "league": 1}
    x = {t: torch.randn(sizes[t], width, generator=g) for t in NODE_TYPES}
    edges = {}
    for rel in METADATA_MP[1]:
        s, _, d = rel
        n = 12
        edges[rel] = torch.stack([torch.randint(0, sizes[s], (n,), generator=g), torch.randint(0, sizes[d], (n,), generator=g)])
    return x, edges


class FastHGT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.conv = FastHGTConv(16, METADATA_MP, heads=2)
        with torch.no_grad():  # non-trivial relation priors and skip gates
            for p in self.conv.conv.p_rel.values():
                p.uniform_(0.5, 1.5)
            for p in self.conv.conv.skip.values():
                p.uniform_(-1.0, 1.0)

    def test_equals_pyg_hgtconv_without_weights(self):
        x, edges = random_graph()
        ours = self.conv(x, edges)
        ref = self.conv.conv(x, edges)
        self.assertEqual(set(ours), set(ref))
        for t in ours:
            torch.testing.assert_close(ours[t], ref[t], rtol=1e-5, atol=1e-6)

    def test_log_weight_of_two_equals_a_duplicated_edge(self):
        x, edges = random_graph(seed=1)
        rel = ("player", "in", "slot")
        logw = {rel: torch.zeros(edges[rel].shape[1])}
        logw[rel][3] = torch.log(torch.tensor(2.0))
        weighted = self.conv(x, edges, logw)
        doubled = dict(edges)
        doubled[rel] = torch.cat([edges[rel], edges[rel][:, 3:4]], 1)
        ref = self.conv(x, doubled)
        for t in weighted:
            torch.testing.assert_close(weighted[t], ref[t], rtol=1e-5, atol=1e-6)


def tiny(variant="m2", seed=0):
    """A small float64 entity model with every parameter drawn at random."""
    torch.manual_seed(seed)
    sizes = {"player": 3, "slot": 2, "team": 1, "role": 2, "team_type": 2, "league": 1}
    model = EntityModel(sizes, 4, variant, 44).double()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()
    return model, sizes


def lam_table(seed=1):
    g = torch.Generator().manual_seed(seed)
    lam = {"d": 1.0}
    for level in ("player", "slot", "physique", "type"):
        for t in CELLS:
            lam[(level, t)] = float(torch.rand(1, generator=g)) + 0.5
    return lam


def fake_data(term, cell, player, slot, role, style, phys):
    """One target whose cell of the given term is set (others 0), with the candidate (player, slot, role, style)."""
    one = lambda v: torch.tensor([[v, v]])
    cells = {t: torch.tensor([cell if t == term else 0]) for t in CELLS}
    return SimpleNamespace(player=one(player), slot=one(slot), role=one(role), team_type=one(style),
                           phys=phys, cell=cells, opp_on=torch.tensor([True]), device=torch.device("cpu"),
                           offset=torch.zeros(1, 2, dtype=torch.float64))


class Ridge(unittest.TestCase):
    def test_invariant_to_rescaling_the_tables_against_the_vectors(self):
        model, _ = tiny()
        lam = lam_table()
        before = float(model.ridge(lam))
        with torch.no_grad():
            for A in list(model.A_s.values()) + list(model.A_p.values()):
                A.mul_(3.0)
            for w in model.penalised_parameters:
                w.div_(3.0)
        self.assertAlmostEqual(float(model.ridge(lam)), before, places=9)

    def test_equals_the_squared_gradient_effects_of_the_forward_pass(self):
        """For every term t (the model restricted to t), the ridge equals 1/2 sum lam <d score / d v, v>^2 over the
        penalised vectors v and the cells of t, the gradients taken through the forward pass."""
        model, sizes = tiny()
        lam = lam_table()
        all_s, all_p = list(model.terms_s), list(model.terms_p)
        for t in CELLS:
            model.terms_s = [x for x in all_s if x == t]
            model.terms_p = [x for x in all_p if x == t]
            ref = 0.0
            for z in range(CELLS[t]):
                for player in range(sizes["player"]):
                    for slot in range(sizes["slot"]):
                        role, style = slot % sizes["role"], (player + slot) % sizes["team_type"]
                        phys = torch.zeros(sizes["player"], 5, dtype=torch.float64)
                        D = fake_data(t, z, player, slot, role, style, phys)
                        score = model(D, torch.tensor([0]))[0, 0]
                        grads = torch.autograd.grad(score, [model.r_player.weight, model.r_slot.weight, model.role.weight,
                                                            model.style_role.weight], allow_unused=True)
                        g_player, g_slot = grads[0][player], grads[1][slot]
                        # each player and slot vector once per cell: player rows over slot 0, slot rows over player 0
                        if slot == 0:
                            ref += lam[("player", t)] * float(g_player @ model.r_player.weight[player]) ** 2
                        if player == 0:
                            ref += lam[("slot", t)] * float(g_slot @ model.r_slot.weight[slot]) ** 2
                # the type level and the physique columns, through a candidate that uses them
                for role in range(sizes["role"]):
                    D = fake_data(t, z, 0, 0, role, 0, torch.zeros(sizes["player"], 5, dtype=torch.float64))
                    g = torch.autograd.grad(model(D, torch.tensor([0]))[0, 0], model.role.weight)[0][role]
                    ref += lam[("type", t)] * float(g @ model.role.weight[role]) ** 2
                for k in range(sizes["team_type"] * sizes["role"]):
                    style, role = divmod(k, sizes["role"])
                    D = fake_data(t, z, 0, 0, role, style, torch.zeros(sizes["player"], 5, dtype=torch.float64))
                    g = torch.autograd.grad(model(D, torch.tensor([0]))[0, 0], model.style_role.weight)[0][k]
                    ref += lam[("type", t)] * float(g @ model.style_role.weight[k]) ** 2
                for j in range(5):
                    phys = torch.zeros(sizes["player"], 5, dtype=torch.float64)
                    phys[0, j] = 1.0
                    D = fake_data(t, z, 0, 0, 0, 0, phys)
                    g = torch.autograd.grad(model(D, torch.tensor([0]))[0, 0], model.physique.weight)[0][:, j]
                    ref += lam[("physique", t)] * float(g @ model.physique.weight[:, j]) ** 2
            self.assertAlmostEqual(float(model.ridge(lam)), 0.5 * ref, places=8, msg=t)


if __name__ == '__main__':
    unittest.main()
