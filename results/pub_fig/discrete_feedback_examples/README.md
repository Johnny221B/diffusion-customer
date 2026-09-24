# Discrete-feedback examples

All four labeled images are 496 x 566 PNG files used to illustrate discrete
feedback. Each has a title above the complete original 496 x 496 image. Panels (b)--(d) come from the same 228-word x 40-seed image pool.

| Panel | File | Role | DreamSim distance to reference |
|---|---|---|---:|
| (a) | `a_reference_red.png` | Reference generated from `red` | 0 |
| (b) | `b_canvas_seed34_d0.6205.png` | Competitor / decision threshold | 0.620474 |
| (c) | `c_below_threshold_red_seed34_d0.4340.png` | Below threshold (`hard win = 1`) | 0.434018 |
| (d) | `d_above_threshold_mosaic_seed34_d0.6973.png` | Above threshold (`hard win = 0`) | 0.697317 |

The pool image in panel (c) is a separate `red` render; it is not the
reference image itself.

Layout: reference (top left), competitor (top right), closer candidate (bottom left), farther candidate (bottom right). Closer/farther refers to DreamSim distance to the reference relative to the competitor, not a guaranteed sampled binary response.

Unlabeled originals are preserved in `outputs/discrete_feedback_examples_originals/`. Reproduce labels with `scripts/133_label_discrete_feedback_examples.py`.
