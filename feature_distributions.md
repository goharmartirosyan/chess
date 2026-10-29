# Feature Distribution Report

Input: `bad_moves_depth18.csv`

Rows read: 862

Rows analyzed (`initial_damage > 20`): 216

This is descriptive analysis, not a causal estimate. Numeric relationships use complete cases only.

## Persistence Target Distributions

| Target | N | Min | P05 | P25 | Median | Mean | P75 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence ratio at 6 plies | 202 | -2.1765 | -0.9270 | 0.3390 | 1.0145 | 0.8020 | 1.2269 | 1.9567 | 3.4137 |
| Remaining win-chance damage at 6 plies | 202 | -56.8417 | -38.6597 | 9.4980 | 30.7795 | 25.9938 | 42.6608 | 70.2883 | 85.7802 |

`persistence_ratio_6ply` is dimensionless: `1.0` means all immediate damage remains after six plies, `0.0` means it was fully recovered, and values above `1.0` mean the position worsened further for the original mistake-maker.

## Numeric Relationships

Pearson captures linear association; Spearman captures monotonic association. Quintiles divide each feature's observed values into equal-sized groups and report the target distribution within each group.

### Persistence ratio at 6 plies

| Feature | Complete rows | Pearson r | Spearman r |
| --- | ---: | ---: | ---: |
| Opponent Elo | 202 | -0.0144 | -0.0360 |
| Rating difference (mistake-maker minus opponent) | 202 | -0.1860 | -0.1651 |
| Opponent clock before move (seconds) | 202 | 0.1108 | 0.0295 |
| Mistake-maker clock before move (seconds) | 202 | 0.0582 | -0.0092 |
| Immediate win-chance drop (percentage points) | 202 | -0.0643 | -0.1332 |
| Move number | 202 | -0.0565 | -0.0033 |
| Material balance (mistake-maker POV, cp) | 202 | 0.0797 | 0.1221 |
| Legal moves / branching factor | 202 | -0.0068 | 0.0236 |
| Best-to-second-best evaluation gap (cp) | 202 | -0.0335 | -0.0268 |

Quintile target distributions are in the accompanying detailed CSV.

### Remaining win-chance damage at 6 plies

| Feature | Complete rows | Pearson r | Spearman r |
| --- | ---: | ---: | ---: |
| Opponent Elo | 202 | -0.0193 | -0.0548 |
| Rating difference (mistake-maker minus opponent) | 202 | -0.1877 | -0.1292 |
| Opponent clock before move (seconds) | 202 | 0.0670 | 0.0010 |
| Mistake-maker clock before move (seconds) | 202 | 0.0276 | -0.0235 |
| Immediate win-chance drop (percentage points) | 202 | 0.2771 | 0.2842 |
| Move number | 202 | -0.0055 | 0.0440 |
| Material balance (mistake-maker POV, cp) | 202 | 0.1015 | 0.1522 |
| Legal moves / branching factor | 202 | -0.0376 | 0.0245 |
| Best-to-second-best evaluation gap (cp) | 202 | 0.1357 | 0.0206 |

Quintile target distributions are in the accompanying detailed CSV.

## Categorical Relationships

### Persistence ratio at 6 plies

| Feature | Group | N | Median | Mean | P25 | P75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Game phase | opening | 128 | 1.0145 | 0.8672 | 0.4240 | 1.3244 |
| Game phase | middlegame | 57 | 1.0122 | 0.6732 | 0.0800 | 1.1478 |
| Time control | 600+0 | 53 | 1.0145 | 0.9178 | 0.5212 | 1.2386 |
| Time control | 60+0 | 53 | 1.0348 | 0.7261 | 0.0612 | 1.3272 |
| Time control | 180+0 | 49 | 0.9300 | 0.7537 | 0.0789 | 1.2135 |
| Time control | 300+0 | 44 | 1.0366 | 0.7678 | 0.0755 | 1.1366 |

### Remaining win-chance damage at 6 plies

| Feature | Group | N | Median | Mean | P25 | P75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Game phase | opening | 128 | 30.7795 | 27.1820 | 12.2943 | 42.6758 |
| Game phase | middlegame | 57 | 28.8981 | 23.5028 | 3.1257 | 42.1524 |
| Time control | 600+0 | 53 | 32.5444 | 27.5257 | 18.5016 | 38.4032 |
| Time control | 60+0 | 53 | 30.3897 | 24.2854 | 3.7965 | 43.6119 |
| Time control | 180+0 | 49 | 28.0109 | 24.6266 | 2.4363 | 45.1882 |
| Time control | 300+0 | 44 | 30.0486 | 26.9220 | 2.9663 | 46.2878 |

