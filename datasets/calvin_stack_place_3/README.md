# CALVIN Stack+Place 3

Audited destination-conditioned specialist dataset containing `stack_block`,
`place_in_drawer`, and `place_in_slider`. See
[`datasets/calvin_specialists/README.md`](../calvin_specialists/README.md) for
construction rules and commands.

Final audited counts:

| Split | Raw annotations | Overlap clusters | Retained episodes | Frames |
|---|---:|---:|---:|---:|
| training | 454 | 370 | 325 | 14,527 |
| validation | 88 | 65 | 60 | 2,626 |

Training contains 69 stable stack episodes covering all six ordered
source-to-destination color pairs. Validation contains 13 stable stack episodes.
Only the unique destination is outlined; the held source is left unchanged.
