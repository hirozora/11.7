import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from hypothesis_statistics import possessions, quick_attack


def main():
    rows = []

    def add(pid, sec, action, x, y, ex, ey, start="interception"):
        rows.append({"match_id": 1, "possession_id": pid, "team_id": 1, "possession_start": start, "event_sec": sec,
                     "action": action, "x": x, "y": y, "end_x": ex, "end_y": ey})

    add(0, 100.0, "touch", 60, 50, 60, 50)          # touch is not an effective action
    add(0, 101.0, "short_pass", 60, 50, 70, 50)
    add(0, 104.0, "long_pass", 70, 50, 90, 50)      # ends in the box at 4 s: quick
    add(1, 200.0, "short_pass", 60, 50, 62, 50)
    add(1, 201.0, "short_pass", 62, 50, 64, 50)
    add(1, 202.0, "carry", 64, 50, 70, 50)
    add(1, 203.0, "shot", 80, 50, 100, 50)          # fourth effective action: not quick
    add(2, 300.0, "short_pass", 60, 50, 62, 50, "opp_pass_lost")
    add(2, 311.0, "shot", 80, 50, 100, 50, "opp_pass_lost")      # after 10 s: not quick
    add(3, 400.0, "clearance", 70, 50, 90, 50, "opp_clearance")  # clearance is not effective: not quick
    add(4, 500.0, "shot", 40, 50, 100, 50)          # starts in own half: excluded
    add(5, 600.0, "shot", 70, 50, 100, 50, "restart_out")        # set piece: excluded
    add(6, 700.0, "take_on", 90, 50, 92, 10)        # starts in the box
    add(6, 709.9, "shot", 92, 40, 100, 50)          # shot within 10 s: quick and shot
    add(7, 800.0, "cross", 85, 10, 95, 50, "duel_won")           # wide start, cross ends in the box: quick
    A = pd.DataFrame(rows)
    P = possessions(A)
    quick, shot = quick_attack(A, P)
    assert list(P["possession_id"]) == [0, 1, 2, 3, 6, 7]
    assert list(P["active"]) == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert list(P["in_box"]) == [False, False, False, False, True, False]
    assert list(quick) == [True, False, False, False, True, True]
    assert list(shot) == [False, False, False, False, True, False]
    print("quick-attack checks passed")


if __name__ == "__main__":
    main()
