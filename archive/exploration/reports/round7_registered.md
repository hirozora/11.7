# Round 7 (development run, England training matches): registered combinations and tree-leaf scenes

Protocol: PROTOCOL_round7.md (frozen before any computation, hash in reports/protocol_round7.sha256).
Development results are 'first registered test on development data'; main conclusions come from the confirmatory run.

## Registered combinations: population layer (cluster-robust Wald, Holm over 8 tests at 0.05)

| test | Wald | df | p | Holm | key effects (odds ratio per SD or vs reference, 95% CI) |
|---|---|---|---|---|---|
| P1b: urgency -> team long pass | 23.0 | 1 | 1.64e-06 | reject | U 1.069 [1.040, 1.099] |
| P1c: urgency -> team cross | 138.0 | 1 | 7.44e-32 | reject | U 1.199 [1.163, 1.236] |
| P2a: score x steps (0-2) -> forward (df 4) | 2.5 | 4 | 6.52e-01 | - | leading 0.993 [0.946, 1.042]; trailing 1.013 [0.967, 1.061]; leading x steps1 1.016 [0.957, 1.080]; leading x steps2 0.988 [0.928, 1.053]; trailing x steps1 0.969 [0.910, 1.031]; trailing x steps2 1.011 [0.944, 1.083] |
| P2b: score x steps (0-2) -> back pass (df 4) | 14.2 | 4 | 6.55e-03 | reject | leading 0.875 [0.817, 0.937]; trailing 1.021 [0.964, 1.081]; leading x steps1 1.109 [1.035, 1.187]; leading x steps2 1.135 [1.043, 1.235]; trailing x steps1 0.996 [0.922, 1.076]; trailing x steps2 0.929 [0.861, 1.003] |
| P5: trailing after 75' -> team cross | 94.5 | 1 | 2.48e-22 | reject | trailing_late 1.574 [1.436, 1.725] |
| P1a: urgency x position group -> who performs (joint, df 6) | 75.7 | 6 | 2.71e-14 | reject | U 0.973 [0.960, 0.987]; U x GK 1.058 [1.026, 1.091]; U x CB 1.098 [1.066, 1.130]; U x FB 1.048 [1.023, 1.073]; U x DM 1.039 [1.016, 1.062]; U x WIDE 1.038 [1.013, 1.063]; U x FW 0.958 [0.935, 0.982] |
| P4: fatigue x position group -> who performs (df 6) | 13.6 | 6 | 3.48e-02 | - | F 0.973 [0.953, 0.993]; F x GK 1.092 [0.896, 1.332]; F x CB 1.021 [0.973, 1.070]; F x FB 1.069 [1.020, 1.121]; F x DM 1.000 [0.964, 1.038]; F x WIDE 0.994 [0.961, 1.028]; F x FW 1.028 [0.997, 1.060] |
| P3: opponent event x position group -> who takes over, steps 0-1 (df 24) | 2355.5 | 24 | 0.00e+00 | reject | duel 1.128 [1.100, 1.157]; interception 1.006 [0.968, 1.046]; clearance 0.815 [0.747, 0.890]; other 0.965 [0.928, 1.005]; steps1 0.989 [0.984, 0.993]; duel x GK 0.612 [0.568, 0.660]; duel x CB 0.715 [0.686, 0.744]; duel x FB 0.752 [0.716, 0.789] |

P3 has 24 interaction terms; the strongest are listed in the JSON.

## Registered combinations: entity layer (score statistic, 1000 match-wise label permutations, Holm over 11 at 0.05)

Floor: planted entity effects varying between matches; power at the Bonferroni level 0.05 / 11; tau80 = smallest planted
tau with power >= 0.8 (slopes: log-odds per SD of the index).

| test | rows | tau | typical OR | p | Holm | floor power at tau 0.1 / 0.2 / 0.3 / 0.4 | tau80 | forward gain (x1000 nats) [CI] |
|---|---|---|---|---|---|---|---|---|
| E1a: slot slopes on urgency (who) | 3311250 | 0.018 | 1.018 | 0.0639 | - | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | - |
| E1b: player slopes on urgency (who, beyond slot) | 3311250 | 0.019 | 1.019 | 0.0190 | - | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | - |
| E1c: team slopes on urgency (long pass) | 303599 | 0.044 | 1.045 | 0.0470 | - | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | - |
| E1d: team slopes on urgency (cross) | 303599 | 0.000 | 1.000 | 0.7123 | - | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | - |
| E2a: team x score state, steps 0-2 (forward) | 156626 | 0.025 | 1.025 | 0.1339 | - | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | - |
| E2b: team x score state, steps 0-2 (back pass) | 156626 | 0.004 | 1.004 | 0.4615 | - | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | - |
| E3a: slot x opponent event, steps 0-1 (who) | 1290027 | 0.053 | 1.055 | 0.0010 | reject | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | +0.018 [+0.004, +0.032] (tau 0.051) |
| E3b: player x opponent event, steps 0-1 (who, beyond slot) | 1290027 | 0.080 | 1.083 | 0.0010 | reject | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | +0.080 [+0.057, +0.103] (tau 0.074) |
| E4a: player slopes on fatigue (who, beyond slot) | 3311250 | 0.019 | 1.019 | 0.0180 | - | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | - |
| E4b: slot slopes on fatigue (who) | 3311250 | 0.034 | 1.035 | 0.0010 | reject | 1.0 / 1.0 / 1.0 / 1.0 | 0.1 | +0.015 [+0.002, +0.030] (tau 0.031) |
| E5: team x trailing after 75' (cross) | 303599 | 0.000 | 1.000 | 0.5225 | - | 0.1 / 0.9 / 1.0 / 1.0 | 0.2 | - |

## Tree-leaf scenes (exploratory)

One shared tau per level over entity x leaf cells; 200 permutations. Leaves fitted on the first 19 matches of each team.

| task / level / K | p | tau | forward gain [CI] | leaves with fewer than 50 matches |
|---|---|---|---|---|
| team|long_pass|K8 | 0.005 | 0.054 | -0.042 [-0.102, +0.018] | 0 |
| team|long_pass|K16 | 0.005 | 0.077 | -0.001 [-0.070, +0.058] | 0 |
| team|long_pass|K32 | 0.005 | 0.093 | -0.026 [-0.110, +0.056] | 0 |
| team|cross|K8 | 0.010 | 0.100 | +0.019 [+0.004, +0.033] | 0 |
| team|cross|K16 | 0.005 | 0.126 | +0.040 [-0.004, +0.086] | 0 |
| team|cross|K32 | 0.015 | 0.126 | +0.031 [-0.000, +0.064] | 0 |
| team|forward|K8 | 0.005 | 0.048 | +0.006 [-0.093, +0.101] | 0 |
| team|forward|K16 | 0.005 | 0.063 | +0.013 [-0.101, +0.128] | 0 |
| team|forward|K32 | 0.005 | 0.068 | +0.004 [-0.120, +0.126] | 0 |
| team|back_pass|K8 | 0.005 | 0.046 | +0.091 [+0.017, +0.166] | 0 |
| team|back_pass|K16 | 0.005 | 0.056 | +0.098 [+0.025, +0.173] | 0 |
| team|back_pass|K32 | 0.005 | 0.074 | +0.135 [+0.047, +0.227] | 0 |
| slot|who|K8 | 0.005 | 0.033 | +0.006 [-0.002, +0.015] | 0 |
| player|who|K8 | 0.005 | 0.044 | +0.006 [-0.002, +0.014] | 0 |
| slot|who|K16 | 0.005 | 0.061 | +0.065 [+0.037, +0.094] | 0 |
| player|who|K16 | 0.005 | 0.080 | +0.107 [+0.071, +0.143] | 0 |
| slot|who|K32 | 0.005 | 0.071 | +0.140 [+0.091, +0.195] | 0 |
| player|who|K32 | 0.005 | 0.093 | +0.171 [+0.126, +0.220] | 0 |

Floor (K = 16, effect planted in one random leaf; power at 0.05 for tau 0.1 / 0.2 / 0.3 / 0.4): team|long_pass|K16: 0.0 / 0.0 / 0.1 / 0.1; slot|who|K16: 0.0 / 0.1 / 1.0 / 1.0

Leaves with the largest excess (observed - permuted leaf statistic), K = 16:

- team|long_pass|K16 (p 0.005):
  - excess +30.2, rows 31433, matches 266, rate 8.8%: not possession_start=interception & score_diff <= 2.5 & steps > 1.5 & 22.5 < x <= 43.5 & 9.5 < y <= 89.5
  - excess +17.8, rows 8462, matches 266, rate 45.1%: not phase=first & not possession_start=interception & x <= 15.5
  - excess +15.0, rows 3963, matches 266, rate 14.8%: opp5=other & not possession_start=interception & score_diff <= 2.5 & steps <= 1.5 & 22.5 < x <= 43.5 & 9.5 < y <= 89.5
- team|cross|K16 (p 0.005):
  - excess +26.6, rows 5298, matches 266, rate 42.8%: steps > 2.5 & x > 91.5
  - excess +19.0, rows 14814, matches 266, rate 0.8%: 65.5 < x <= 68.5
  - excess +13.2, rows 7795, matches 266, rate 3.5%: 68.5 < x <= 74.5 & y <= 24.5
- team|forward|K16 (p 0.005):
  - excess +68.7, rows 5908, matches 266, rate 64.3%: not phase=first & x <= 16.5 & 36.5 < y <= 63.5
  - excess +33.5, rows 3112, matches 266, rate 48.9%: not phase=first & x <= 16.5 & y > 63.5
  - excess +17.0, rows 59142, matches 266, rate 22.7%: 16.5 < x <= 33.5
- team|back_pass|K16 (p 0.005):
  - excess +33.9, rows 55635, matches 266, rate 12.6%: 64.5 < x <= 93.5 & 13.5 < y <= 84.5
  - excess +12.9, rows 13991, matches 266, rate 28.7%: not phase=first & not opp5=duel & x > 34.5 & y <= 13.5
  - excess +12.4, rows 14830, matches 266, rate 23.1%: opp5=duel & not phase=first & x > 34.5 & y <= 13.5
- slot|who|K16 (p 0.005):
  - excess +224.9, rows 397832, matches 266, rate 11.0%: not fine_role=GK & not fine_role=backmid/centre & not fine_role=front/centre & x > 15.5 & y <= 16.5
  - excess +220.8, rows 1818681, matches 266, rate 9.8%: not fine_role=GK & not fine_role=backmid/centre & not fine_role=front/centre & x > 15.5 & y > 16.5
  - excess +150.9, rows 114561, matches 266, rate 4.7%: not fine_role=GK & steps > 0.5 & x <= 15.5
- player|who|K16 (p 0.005):
  - excess +353.9, rows 397832, matches 266, rate 11.0%: not fine_role=GK & not fine_role=backmid/centre & not fine_role=front/centre & x > 15.5 & y <= 16.5
  - excess +316.5, rows 146809, matches 266, rate 9.1%: not fine_role=GK & steps <= 0.5 & x <= 15.5
  - excess +274.2, rows 1818681, matches 266, rate 9.8%: not fine_role=GK & not fine_role=backmid/centre & not fine_role=front/centre & x > 15.5 & y > 16.5
