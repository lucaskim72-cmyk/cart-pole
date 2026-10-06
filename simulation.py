# -*- coding: utf-8 -*-
"""CartPole epsilon-greedy simulation with random side collisions."""

from __future__ import annotations

import io
import math
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

IMPACT_FORCES = {"약": 0.5, "중": 1.5, "강": 3.0}
NATIVE_W = 600.0
NATIVE_H = 400.0
IMPACT_FRAMES = 22
BODY_SIZE = {"약": (36.0, 16.0), "중": (50.0, 22.0), "강": (64.0, 26.0)}
BALL_RADIUS = {"약": 8.0, "중": 13.0, "강": 18.0}
BUMPER_COLOR = {"약": (78, 138, 156), "중": (188, 112, 36), "강": (168, 46, 40)}
DUST_PATTERN = (
    (-1.4, -0.35, 3.4),
    (1.1, -0.55, 2.6),
    (-0.4, -0.9, 2.2),
    (1.6, -0.2, 3.0),
    (-1.8, -0.15, 2.4),
    (0.5, -0.7, 1.8),
)


class DisturbedCartPoleEnv(gym.Wrapper):
    """Apply a random lateral impact after each physics step."""

    def __init__(self, env: gym.Env, collision_prob: float, rng: np.random.Generator):
        super().__init__(env)
        self.collision_prob = float(collision_prob)
        self.rng = rng

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        collision = None

        if self.rng.random() < self.collision_prob:
            level = str(self.rng.choice(list(IMPACT_FORCES.keys())))
            force = IMPACT_FORCES[level]
            direction = int(self.rng.choice([-1, 1]))
            state = np.array(self.env.unwrapped.state, dtype=np.float64)
            state[1] += force * direction
            state[3] += force * direction * 1.5
            self.env.unwrapped.state = state
            observation = np.array(state, dtype=np.float32)
            collision = {
                "level": level,
                "direction": direction,
                "force": force,
            }

        info = dict(info)
        info["collision"] = collision
        return observation, reward, terminated, truncated, info


def get_best_action(observation: np.ndarray) -> int:
    """Push toward the side the pole is leaning away from."""
    pole_angle = float(observation[2])
    return 1 if pole_angle < 0 else 0


def _rotate(vx: float, vy: float, angle: float) -> tuple[float, float]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return cosine * vx - sine * vy, sine * vx + cosine * vy


def _to_image(px: float, py: float, sx: float, sy: float) -> tuple[float, float]:
    return px * sx, (NATIVE_H - 1.0 - py) * sy


def _scene(env: gym.Env, observation: np.ndarray, frame_shape: tuple[int, ...]) -> dict[str, float]:
    height, width = frame_shape[:2]
    sx, sy = width / NATIVE_W, height / NATIVE_H
    x_threshold = float(env.unwrapped.x_threshold)
    cartx = float(observation[0]) / (x_threshold * 2.0) * NATIVE_W + NATIVE_W / 2.0
    cart_ix, cart_iy = _to_image(cartx, 100.0, sx, sy)
    return {
        "sx": sx,
        "sy": sy,
        "cartx": cartx,
        "theta": float(observation[2]),
        "cart_ix": cart_ix,
        "cart_iy": cart_iy,
    }


def _pole_hit(scene: dict[str, float], fraction: float = 0.62) -> tuple[float, float]:
    span = 120.0 * fraction
    rx, ry = _rotate(0.0, span, -scene["theta"])
    return _to_image(scene["cartx"] + rx, 107.5 + ry, scene["sx"], scene["sy"])


def _gap_outside(progress: float, travel: float) -> tuple[float, float]:
    """Distance still left before contact, and how hard the body is compressed."""
    if progress < 0.50:
        blend = progress / 0.50
        ease = blend * blend * (3.0 - 2.0 * blend)
        return travel * (1.0 - ease), 0.0
    if progress < 0.66:
        blend = (progress - 0.50) / 0.16
        return 0.0, math.sin(blend * math.pi)
    blend = (progress - 0.66) / 0.34
    return travel * (blend ** 1.3), 0.0


def _paint_wheel(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, phase: float, alpha: int) -> None:
    tire = (22, 24, 28, alpha)
    rim = (186, 190, 196, alpha)
    hub = (58, 62, 68, alpha)
    spoke = (36, 38, 44, alpha)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tire)
    inner = radius * 0.62
    draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=rim)
    hub_r = radius * 0.22
    draw.ellipse((x - hub_r, y - hub_r, x + hub_r, y + hub_r), fill=hub)
    spoke_len = radius * 0.72
    width = max(1, int(radius * 0.14))
    for index in range(3):
        angle = phase + index * math.pi / 3.0
        draw.line(
            (x, y, x + math.cos(angle) * spoke_len, y + math.sin(angle) * spoke_len),
            fill=spoke,
            width=width,
        )


def _paint_cart(
    draw: ImageDraw.ImageDraw,
    left: float,
    right: float,
    center_y: float,
    body_h: float,
    level: str,
    squash: float,
    wheel_phase: float,
    faces_right: bool,
    alpha: int,
    draw_cargo: bool,
) -> None:
    body_w = right - left
    top = center_y - body_h / 2.0
    bottom = center_y + body_h / 2.0
    radius = max(2.0, min(body_h * 0.28, body_w * 0.2))
    shadow_w = body_w * 0.92
    shadow_h = max(3.0, body_h * 0.22)
    shadow_cx = (left + right) / 2.0
    shadow_cy = bottom + body_h * 0.28
    shadow_alpha = max(0, min(255, int(alpha * 0.28)))
    draw.ellipse(
        (
            shadow_cx - shadow_w / 2.0,
            shadow_cy - shadow_h / 2.0,
            shadow_cx + shadow_w / 2.0,
            shadow_cy + shadow_h / 2.0,
        ),
        fill=(40, 44, 52, shadow_alpha),
    )

    wheel_r = body_h * 0.48
    wheel_y = center_y + body_h * 0.34
    inset = body_w * 0.24
    _paint_wheel(draw, left + inset, wheel_y, wheel_r, wheel_phase, alpha)
    _paint_wheel(draw, right - inset, wheel_y, wheel_r, wheel_phase, alpha)

    shell = (58, 64, 72, alpha) if level == "강" else (86, 93, 102, alpha)
    roof = (132, 140, 150, alpha) if level != "강" else (108, 114, 122, alpha)
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=shell, outline=(32, 36, 42, alpha), width=1)
    roof_bottom = top + body_h * 0.46
    draw.rounded_rectangle((left + 1.0, top + 1.0, right - 1.0, roof_bottom), radius=radius * 0.8, fill=roof)
    draw.line(
        (left + body_w * 0.12, top + 2.2, right - body_w * 0.12, top + 2.2),
        fill=(214, 220, 226, alpha),
        width=max(1, int(body_h * 0.08)),
    )

    panel_w = body_w * 0.28
    panel_h = body_h * 0.28
    panel_cx = (left + right) / 2.0 + (-body_w * 0.08 if faces_right else body_w * 0.08)
    draw.rounded_rectangle(
        (
            panel_cx - panel_w / 2.0,
            top + body_h * 0.16,
            panel_cx + panel_w / 2.0,
            top + body_h * 0.16 + panel_h,
        ),
        radius=2,
        fill=(196, 206, 214, alpha),
    )

    thick = max(4.0, body_h * 0.36) * (1.0 - 0.45 * squash)
    pad = BUMPER_COLOR[level] + (alpha,)
    if faces_right:
        draw.rounded_rectangle((right - thick, top + body_h * 0.16, right + thick * 0.35, bottom - body_h * 0.16), radius=2, fill=pad)
    else:
        draw.rounded_rectangle((left - thick * 0.35, top + body_h * 0.16, left + thick, bottom - body_h * 0.16), radius=2, fill=pad)

    if draw_cargo and level == "강":
        crate_w = body_w * 0.40
        crate_h = body_h * 0.62
        crate_cx = (left + right) / 2.0 + (-body_w * 0.12 if faces_right else body_w * 0.12)
        crate_top = top - crate_h * 0.78
        draw.rounded_rectangle(
            (crate_cx - crate_w / 2.0, crate_top, crate_cx + crate_w / 2.0, crate_top + crate_h),
            radius=2,
            fill=(122, 84, 52, alpha),
        )
        draw.line(
            (crate_cx - crate_w / 2.0, crate_top + crate_h * 0.5, crate_cx + crate_w / 2.0, crate_top + crate_h * 0.5),
            fill=(92, 60, 36, alpha),
            width=1,
        )


def _paint_sphere(draw: ImageDraw.ImageDraw, x: float, y: float, rx: float, ry: float, alpha: int) -> None:
    draw.ellipse(
        (x - rx * 0.85, y + ry * 0.55, x + rx * 0.9, y + ry * 1.25),
        fill=(30, 32, 36, max(0, int(alpha * 0.28))),
    )
    draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=(48, 52, 58, alpha))
    draw.ellipse(
        (x - rx * 0.78, y - ry * 0.82, x + rx * 0.42, y + ry * 0.35),
        fill=(118, 124, 132, alpha),
    )
    draw.ellipse(
        (x - rx * 0.55, y - ry * 0.68, x - rx * 0.12, y - ry * 0.18),
        fill=(244, 247, 250, alpha),
    )
    draw.arc(
        (x - rx, y - ry, x + rx, y + ry),
        start=200,
        end=340,
        fill=(24, 26, 30, alpha),
        width=max(1, int(min(rx, ry) * 0.12)),
    )


def _spawn_debris(effect: dict[str, Any], x: float, y: float, from_sign: float) -> None:
    amount = 8 if effect["level"] == "강" else 5
    particles = []
    for index, (jx, jy, radius) in enumerate(DUST_PATTERN[:amount]):
        particles.append(
            {
                "x": x + jx * 3.0,
                "y": y + 4.0,
                "vx": from_sign * (-0.35 - index * 0.08) + jx * 0.45,
                "vy": jy,
                "life": 9 + (index % 3),
                "max_life": 11,
                "radius": radius,
            }
        )
    effect["particles"] = particles
    effect["spawned"] = True


def _draw_debris(draw: ImageDraw.ImageDraw, effect: dict[str, Any]) -> None:
    alive = []
    for particle in effect["particles"]:
        particle["x"] += particle["vx"]
        particle["y"] += particle["vy"]
        particle["vy"] += 0.07
        particle["life"] -= 1
        if particle["life"] <= 0:
            continue
        fade = particle["life"] / particle["max_life"]
        alpha = int(210 * fade)
        radius = particle["radius"] * (0.85 + 0.35 * (1.0 - fade))
        dust = (128, 114, 96, alpha)
        draw.ellipse(
            (
                particle["x"] - radius,
                particle["y"] - radius * 0.65,
                particle["x"] + radius,
                particle["y"] + radius * 0.65,
            ),
            fill=dust,
        )
        alive.append(particle)
    effect["particles"] = alive


def _draw_sparks(draw: ImageDraw.ImageDraw, x: float, y: float, progress: float) -> None:
    if not 0.50 <= progress <= 0.64:
        return
    fade = 1.0 - (progress - 0.50) / 0.14
    alpha = int(210 * fade)
    flash = 4.0 + 3.0 * fade
    draw.ellipse((x - flash, y - flash * 0.65, x + flash, y + flash * 0.65), fill=(255, 232, 180, alpha))
    core = flash * 0.45
    draw.ellipse((x - core, y - core, x + core, y + core), fill=(255, 250, 236, alpha))


def _draw_effects(
    frame: np.ndarray,
    effects: list[dict[str, Any]],
    observation: np.ndarray,
    env: gym.Env,
) -> np.ndarray:
    if not effects:
        return frame

    scene = _scene(env, observation, frame.shape)
    sx, sy = scene["sx"], scene["sy"]
    base = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for effect in effects:
        progress = min(1.0, effect["age"] / float(IMPACT_FRAMES - 1))
        outside, squash = _gap_outside(progress, 175.0 * sx)
        from_sign = -float(effect["direction"])
        faces_right = effect["direction"] > 0

        if effect["target"] == "pole":
            hit_x, hit_y = _pole_hit(scene)
            radius = BALL_RADIUS[effect["level"]] * (sx + sy) / 2.0
            center_x = hit_x + from_sign * (radius + outside) - from_sign * squash * radius * 0.45
            center_y = hit_y
            rx = radius * (1.0 - 0.28 * squash)
            ry = radius * (1.0 + 0.16 * squash)
            if outside > 18.0 * sx:
                streak = radius * 1.7
                streak_x = center_x + from_sign * streak * 0.55
                draw.ellipse(
                    (streak_x - streak, center_y - ry * 0.55, streak_x + streak * 0.3, center_y + ry * 0.55),
                    fill=(150, 156, 164, 55),
                )
            back_x = center_x + from_sign * rx * 0.72
            back_y = center_y - ry * 0.2
            tail_x = back_x + from_sign * 30.0 * sx
            tail_y = back_y - 16.0 * sy
            draw.line((back_x, back_y, tail_x, tail_y), fill=(62, 66, 72, 220), width=2)
            for step in (0.34, 0.67, 1.0):
                knot_x = back_x + (tail_x - back_x) * step
                knot_y = back_y + (tail_y - back_y) * step
                knot_r = 2.1 * sx
                draw.ellipse(
                    (knot_x - knot_r, knot_y - knot_r, knot_x + knot_r, knot_y + knot_r),
                    fill=(78, 82, 88, 230),
                )
            _paint_sphere(draw, center_x, center_y, rx, ry, 255)
            contact_x = hit_x
            contact_y = hit_y
        else:
            body_w, body_h = BODY_SIZE[effect["level"]]
            body_w *= sx * (1.0 - 0.16 * squash)
            body_h *= sy
            main_half = 25.0 * sx
            face = scene["cart_ix"] + from_sign * main_half
            bumper_x = face + from_sign * outside - from_sign * squash * 7.0 * sx
            if from_sign < 0:
                right = bumper_x
                left = bumper_x - body_w
            else:
                left = bumper_x
                right = bumper_x + body_w
            center_x = (left + right) / 2.0
            center_y = scene["cart_iy"]
            if outside > 22.0 * sx:
                shift = 16.0 * sx
                _paint_cart(
                    draw,
                    left + from_sign * shift,
                    right + from_sign * shift,
                    center_y,
                    body_h,
                    effect["level"],
                    0.0,
                    effect["wheel_phase"],
                    faces_right,
                    70,
                    False,
                )
            _paint_cart(
                draw,
                left,
                right,
                center_y,
                body_h,
                effect["level"],
                squash,
                effect["wheel_phase"],
                faces_right,
                255,
                True,
            )
            wheel_r = max(body_h * 0.48, 1.0)
            if effect["prev_x"] is not None:
                effect["wheel_phase"] += (center_x - effect["prev_x"]) / wheel_r
            effect["prev_x"] = center_x
            contact_x = face
            contact_y = center_y

        if squash > 0.2 and not effect["spawned"]:
            _spawn_debris(effect, contact_x, contact_y, from_sign)
        _draw_sparks(draw, contact_x, contact_y - 2.0, progress)
        _draw_debris(draw, effect)
        effect["age"] += 1

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(composed)


def _frames_to_gif(frames: list[np.ndarray]) -> bytes:
    images = [Image.fromarray(frame) for frame in frames]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=50,
        loop=0,
        optimize=True,
    )
    return buffer.getvalue()


def run_simulation(
    epsilon: float,
    collision_prob: float,
    steps: int,
    seed: int,
    gif_width: int = 480,
) -> dict[str, Any]:
    """Run one epsilon-greedy episode sequence and return a GIF plus logs."""
    rng = np.random.default_rng(int(seed))
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = DisturbedCartPoleEnv(base_env, collision_prob, rng)

    try:
        observation, _info = env.reset(seed=int(seed))
        env.action_space.seed(int(seed))

        frames: list[np.ndarray] = []
        active_effects: list[dict[str, Any]] = []
        explored_flags: list[bool] = []
        actions: list[int] = []
        angles: list[float] = []
        positions: list[float] = []
        rewards: list[float] = []
        collision_levels: list[str | None] = []
        collision_sides: list[str | None] = []
        collision_targets: list[str | None] = []
        fell_flags: list[bool] = []

        for _step in range(int(steps)):
            explored = bool(rng.random() < epsilon)
            if explored:
                action = int(env.action_space.sample())
            else:
                action = get_best_action(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is None:
                raise RuntimeError("렌더링 프레임을 가져오지 못했습니다.")

            frame = np.asarray(frame)
            height, width = frame.shape[:2]
            if width != gif_width:
                target_h = max(1, int(round(height * (gif_width / width))))
                frame = cv2.resize(frame, (gif_width, target_h), interpolation=cv2.INTER_AREA)

            collision = info.get("collision")
            if collision is not None:
                # Positive force pushes the cart to the right, so the other body arrives from the left.
                active_effects.append(
                    {
                        "level": collision["level"],
                        "direction": collision["direction"],
                        "target": "pole" if (_step % 3 == 0) else "cart",
                        "age": 0,
                        "wheel_phase": 0.0,
                        "prev_x": None,
                        "spawned": False,
                        "particles": [],
                    }
                )

            painted = _draw_effects(frame, active_effects, observation, env) if active_effects else frame
            frames.append(painted)
            active_effects = [effect for effect in active_effects if effect["age"] < IMPACT_FRAMES]

            fell = bool(terminated or truncated)
            explored_flags.append(explored)
            actions.append(action)
            angles.append(float(observation[2]))
            positions.append(float(observation[0]))
            rewards.append(float(reward))
            collision_levels.append(collision["level"] if collision else None)
            if collision:
                collision_sides.append("왼쪽에서" if collision["direction"] > 0 else "오른쪽에서")
                collision_targets.append("막대" if (_step % 3 == 0) else "카트")
            else:
                collision_sides.append(None)
                collision_targets.append(None)
            fell_flags.append(fell)

            if fell:
                observation, _info = env.reset()

        explored_arr = np.asarray(explored_flags, dtype=bool)
        fell_arr = np.asarray(fell_flags, dtype=bool)
        level_arr = np.asarray(collision_levels, dtype=object)
        survival_lengths = _episode_lengths(fell_arr, int(steps))

        metrics = {
            "steps": int(steps),
            "falls": int(fell_arr.sum()),
            "explore_count": int(explored_arr.sum()),
            "exploit_count": int((~explored_arr).sum()),
            "explore_rate": float(explored_arr.mean()) if steps else 0.0,
            "collisions": int(np.sum(level_arr != None)),  # noqa: E711
            "collision_weak": int(np.sum(level_arr == "약")),
            "collision_mid": int(np.sum(level_arr == "중")),
            "collision_strong": int(np.sum(level_arr == "강")),
            "total_reward": float(np.sum(rewards)),
            "mean_survival": float(np.mean(survival_lengths)) if survival_lengths else float(steps),
        }

        return {
            "gif": _frames_to_gif(frames),
            "actions": np.asarray(actions, dtype=int),
            "explored": explored_arr,
            "angles": np.asarray(angles, dtype=float),
            "positions": np.asarray(positions, dtype=float),
            "collision_levels": collision_levels,
            "collision_sides": collision_sides,
            "collision_targets": collision_targets,
            "fell": fell_arr,
            "metrics": metrics,
        }
    finally:
        env.close()


def _episode_lengths(fell: np.ndarray, steps: int) -> list[int]:
    lengths: list[int] = []
    start = 0
    for index, did_fall in enumerate(fell):
        if did_fall:
            lengths.append(index - start + 1)
            start = index + 1
    if start < steps:
        lengths.append(steps - start)
    return lengths
