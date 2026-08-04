"""Deterministic mask perturbation and RGB overlay."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskPerturbation:
    max_shift_px: int = 0
    morphology_px: int = 0
    dropout_probability: float = 0.0

    def apply(self, mask, rng):
        import cv2

        result = np.asarray(mask, dtype=np.uint8)
        if self.dropout_probability > 0:
            keep = rng.random_sample(result.shape) >= self.dropout_probability
            result = result * keep.astype(np.uint8)
        if self.morphology_px:
            radius = abs(int(self.morphology_px))
            kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
            op = cv2.MORPH_DILATE if self.morphology_px > 0 else cv2.MORPH_ERODE
            result = cv2.morphologyEx(result, op, kernel)
        if self.max_shift_px:
            dx, dy = rng.randint(
                -self.max_shift_px, self.max_shift_px + 1, size=2
            )
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            result = cv2.warpAffine(
                result,
                transform,
                (result.shape[1], result.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
        return result.astype(bool)


class MaskOverlay:
    DEFAULT_COLORS = {
        "source": (255, 40, 40),
        "destination": (40, 120, 255),
        "target": (255, 200, 40),
    }

    def __init__(
        self,
        alpha=0.45,
        colors=None,
        perturbation=None,
        seed=42,
        outline_width_px=0,
        outline_color=(0, 0, 0),
        outline_colors=None,
        outline_position="center",
        outline_positions=None,
        outline_halo_width_px=0,
        outline_halo_widths=None,
        outline_halo_color=(255, 255, 255),
    ):
        self.alpha = float(alpha)
        self.outline_width_px = int(outline_width_px)
        self.outline_color = np.asarray(outline_color, dtype=np.uint8)
        self.outline_colors = {
            role: np.asarray(value, dtype=np.uint8)
            for role, value in (outline_colors or {}).items()
        }
        self.outline_position = str(outline_position)
        self.outline_positions = {
            role: str(value) for role, value in (outline_positions or {}).items()
        }
        self.outline_halo_width_px = int(outline_halo_width_px)
        self.outline_halo_widths = {
            role: int(value) for role, value in (outline_halo_widths or {}).items()
        }
        self.outline_halo_color = np.asarray(outline_halo_color, dtype=np.uint8)
        if self.outline_color.shape != (3,):
            raise ValueError("outline_color must contain exactly three RGB values")
        if any(value.shape != (3,) for value in self.outline_colors.values()):
            raise ValueError("every role outline color must contain three RGB values")
        valid_positions = {"inside", "center", "outside"}
        if self.outline_position not in valid_positions:
            raise ValueError("outline_position must be inside, center, or outside")
        if any(value not in valid_positions for value in self.outline_positions.values()):
            raise ValueError(
                "every role outline position must be inside, center, or outside"
            )
        if self.outline_halo_width_px < 0:
            raise ValueError("outline_halo_width_px must be >= 0")
        if any(value < 0 for value in self.outline_halo_widths.values()):
            raise ValueError("every role outline halo width must be >= 0")
        if self.outline_halo_color.shape != (3,):
            raise ValueError("outline_halo_color must contain exactly three RGB values")
        if self.outline_width_px < 0:
            raise ValueError("outline_width_px must be >= 0")
        self.colors = dict(self.DEFAULT_COLORS)
        self.colors.update(colors or {})
        self.perturbation = perturbation or MaskPerturbation()
        self.seed = int(seed)

    def apply(self, image, role_masks, frame_index=0):
        result = np.asarray(image, dtype=np.uint8).copy()
        rng = np.random.RandomState(self.seed + int(frame_index))
        for role, mask in role_masks.items():
            mask = self.perturbation.apply(mask, rng)
            if not mask.any():
                continue
            color = np.asarray(
                self.colors.get(role, self.colors["target"]), dtype=np.float32
            )
            blended = (
                (1.0 - self.alpha) * result[mask].astype(np.float32)
                + self.alpha * color
            )
            result[mask] = np.clip(blended, 0, 255).astype(np.uint8)
            if self.outline_width_px:
                import cv2

                radius = self.outline_width_px
                kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
                dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
                eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
                position = self.outline_positions.get(role, self.outline_position)
                if position == "inside":
                    outline = mask & ~eroded
                elif position == "outside":
                    outline = dilated & ~mask
                else:
                    outline = dilated & ~eroded
                halo_width = self.outline_halo_widths.get(
                    role, self.outline_halo_width_px
                )
                if halo_width:
                    halo_radius = radius + halo_width
                    halo_kernel = np.ones(
                        (2 * halo_radius + 1, 2 * halo_radius + 1), np.uint8
                    )
                    halo_dilated = cv2.dilate(
                        mask.astype(np.uint8), halo_kernel
                    ).astype(bool)
                    halo = halo_dilated & ~dilated
                    result[halo] = self.outline_halo_color
                result[outline] = self.outline_colors.get(role, self.outline_color)
        return result
