# CALVIN verifier task status

This file records the current experimental triage so later benchmark rounds do not
repeatedly spend compute on the same tasks. A task moves between groups only after its
Qwen/Oracle results have been reviewed.

## Stable success — freeze pending efficiency and larger-scale validation

- `open_drawer`
- `move_slider_left`
- `place_in_drawer`
- `stack_block`
- `turn_off_led`
- `push_into_drawer`
- `unstack_block`

## Needs more observation


## Repeated failures — pause pending a better verification method

- `turn_on_lightbulb`
- `close_drawer`
- `lift_red_block_table`
- `move_slider_right`
- `turn_on_led`
- `turn_off_lightbulb`
- `place_in_slider`
- `rotate_red_block_right`


## Current exploration batch

No task is currently under active exploration. The two representative rotate tasks from
v17 were reviewed and moved to repeated failures. The other four rotate color/direction
variants were not tested and are not counted in the status totals.
