"""Action targets and population-model context features."""
import numpy as np


STEPS = ["0", "1", "2", "3-5", "6+"]
PASSES = ["short_pass", "long_pass", "cross", "through_ball", "head_pass"]
CAT_CTX = ["zone", "phase", "prev_event", "opp_last_event", "possession_start", "score_state", "time_bin", "venue",
           "opponent_class"]
NUM_CTX = ["x", "y", "steps_code", "poss_start_x", "minute", "score_diff", "minutes_since_goal", "state_duration",
           "poss_share_5", "own_shots_5", "opp_shots_5"]

def derive(A, cfg):
    """Targets and hypothesis levels used by the specs."""
    dx = A["end_x"] - A["x"]
    is_pass = A["action"].isin(PASSES)
    A["t_long_pass"] = (A["action"] == "long_pass").astype(np.int8)
    A["t_cross"] = (A["action"] == "cross").astype(np.int8)
    A["t_back_pass"] = (is_pass & (dx < -cfg["back_pass_dx"])).astype(np.int8)
    A["t_forward"] = (((is_pass | (A["action"] == "carry")) & (dx >= cfg["forward_dx"]))
                      | A["action"].isin(["through_ball", "shot"])).astype(np.int8)
    late = (A["time_bin"] == "after_75") & (A["score_state"] == "trailing")
    A["trailing_late"] = np.where(late, "trailing_late", "other")
    high = ((A["steps_since_change"] == "0") & A["possession_start"].isin(["interception", "duel_won"])
            & (A["poss_start_x"] >= cfg["high_regain_x"]))
    A["high_regain_first"] = np.where(high, "high_regain_first", "other")
    order = sorted(A["match_id"].unique(), key=A.groupby("match_id")["kickoff"].first().get)
    A["era"] = A["match_id"].map({m: int(i >= len(order) // 2) for i, m in enumerate(order)})
    return A

def context(A):
    cols = {}
    for c in CAT_CTX:
        cols[c] = A[c].astype("category").cat.codes.to_numpy()
    A = A.assign(steps_code=A["steps_since_change"].map({s: i for i, s in enumerate(STEPS)}))
    for c in NUM_CTX:
        cols[c] = A[c].to_numpy().astype(float)
    return cols
