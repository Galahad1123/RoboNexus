"""Stable canonical inventory for CALVIN task-D's 34 atomic skills."""

CALVIN_D34_ROUTING = {
    "push": (
        "push_blue_block_left",
        "push_blue_block_right",
        "push_pink_block_left",
        "push_pink_block_right",
        "push_red_block_left",
        "push_red_block_right",
    ),
    "rotate": (
        "rotate_blue_block_left",
        "rotate_blue_block_right",
        "rotate_pink_block_left",
        "rotate_pink_block_right",
        "rotate_red_block_left",
        "rotate_red_block_right",
    ),
    "stack": ("stack_block",),
    "general": (
        "close_drawer",
        "lift_blue_block_drawer",
        "lift_blue_block_slider",
        "lift_blue_block_table",
        "lift_pink_block_drawer",
        "lift_pink_block_slider",
        "lift_pink_block_table",
        "lift_red_block_drawer",
        "lift_red_block_slider",
        "lift_red_block_table",
        "move_slider_left",
        "move_slider_right",
        "open_drawer",
        "place_in_drawer",
        "place_in_slider",
        "push_into_drawer",
        "turn_off_led",
        "turn_off_lightbulb",
        "turn_on_led",
        "turn_on_lightbulb",
        "unstack_block",
    ),
}

CALVIN_D34_SKILL_IDS = frozenset(
    skill_id
    for skill_ids in CALVIN_D34_ROUTING.values()
    for skill_id in skill_ids
)
