#!/usr/bin/env python3
"""Render OpenArm heart-gesture demo video (robot IK + IWER controls overlay) → gofile."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yourdfpy
from PIL import Image, ImageDraw, ImageFont

from teleop_xr.ik.controller import IKController
from teleop_xr.ik.gestures.heart import get_heart_pose_targets
from teleop_xr.ik.robots.openarm import OpenArmRobot
from teleop_xr.ik.solver import PyrokiSolver

OUT_DIR = Path(os.environ.get("DEMO_OUT_DIR", "/home/testpc-3/projects/xr/results"))
FRAME_DIR = OUT_DIR / "vr_heart_frames"
VIDEO_PATH = OUT_DIR / "openarm_vr_heart_demo.mp4"
FPS = 8

# Left / right arm joint chains for stick-figure visualization.
_LEFT_CHAIN = [
    "openarm_body_link0",
    "openarm_left_link0",
    "openarm_left_link1",
    "openarm_left_link2",
    "openarm_left_link3",
    "openarm_left_link4",
    "openarm_left_link5",
    "openarm_left_link6",
    "openarm_left_link7",
    "openarm_left_ee_base_link",
]
_RIGHT_CHAIN = [
    "openarm_body_link0",
    "openarm_right_link0",
    "openarm_right_link1",
    "openarm_right_link2",
    "openarm_right_link3",
    "openarm_right_link4",
    "openarm_right_link5",
    "openarm_right_link6",
    "openarm_right_link7",
    "openarm_right_ee_base_link",
]


def _resolve_chain(urdf: yourdfpy.URDF, names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name in urdf.link_map:
            out.append(name)
        elif name.endswith("_ee_base_link"):
            alt = name.replace("_ee_base_link", "_link7")
            if alt in urdf.link_map:
                out.append(alt)
    return out


def _chain_points(urdf: yourdfpy.URDF, chain: list[str]) -> np.ndarray:
    pts = []
    for link in chain:
        if link not in urdf.link_map:
            continue
        T = urdf.get_transform(link, urdf.base_link)
        pts.append(T[:3, 3])
    return np.asarray(pts, dtype=float)


def _interpolate_heart_configs(robot: OpenArmRobot, controller: IKController, steps: int = 36):
    q = np.array(robot.get_default_config())
    controller.set_mode("ee_absolute")
    heart = get_heart_pose_targets()
    fk0 = robot.forward_kinematics(jnp.asarray(q))
    configs = [q.copy()]

    for step in range(1, steps + 1):
        alpha = step / steps
        blended = {}
        for frame, target in heart.items():
            if frame not in fk0:
                continue
            start_pos = np.asarray(fk0[frame].translation(), dtype=float)
            start_wxyz = np.asarray(fk0[frame].rotation().wxyz, dtype=float)
            goal_pos = np.array(
                [target["position"]["x"], target["position"]["y"], target["position"]["z"]],
                dtype=float,
            )
            goal_wxyz = np.array(
                [
                    target["orientation"]["w"],
                    target["orientation"]["x"],
                    target["orientation"]["y"],
                    target["orientation"]["z"],
                ],
                dtype=float,
            )
            pos = start_pos + alpha * (goal_pos - start_pos)
            wxyz = start_wxyz + alpha * (goal_wxyz - start_wxyz)
            wxyz /= np.linalg.norm(wxyz)
            blended[frame] = {
                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "orientation": {
                    "w": float(wxyz[0]),
                    "x": float(wxyz[1]),
                    "y": float(wxyz[2]),
                    "z": float(wxyz[3]),
                },
            }
        q = np.array(controller.submit_ee_absolute_targets(blended, q))
        configs.append(q.copy())

    controller.set_mode("teleop")
    hold = [q.copy() for _ in range(12)]
    return configs + hold


def _render_robot_frame(
    urdf: yourdfpy.URDF,
    left_chain: list[str],
    right_chain: list[str],
    title: str,
    subtitle: str,
) -> Image.Image:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=18, azim=-58)

    for chain, color in ((left_chain, "#e74c3c"), (right_chain, "#3498db")):
        pts = _chain_points(urdf, chain)
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=color, linewidth=3, markersize=4)

    all_pts = np.vstack([_chain_points(urdf, left_chain), _chain_points(urdf, right_chain)])
    center = all_pts.mean(axis=0)
    span = max(0.45, float(np.max(np.linalg.norm(all_pts - center, axis=1)) * 1.35))
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(max(0.0, center[2] - span), center[2] + span)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=16, pad=16)
    fig.text(0.5, 0.04, subtitle, ha="center", fontsize=11)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _render_iwer_panel(highlight_grip: bool) -> Image.Image:
    w, h = 420, 520
    img = Image.new("RGBA", (w, h), (40, 40, 40, 230))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    draw.text((12, 10), "Meta Quest 3 (IWER)", fill=(255, 255, 255), font=font)
    draw.text((12, 36), "Chrome VR virtual controllers", fill=(180, 180, 180), font=font_sm)

    for idx, (side, x0) in enumerate((("L", 12), ("R", 220))):
        draw.rounded_rectangle((x0, 70, x0 + 188, 500), radius=10, outline=(120, 120, 120), width=2)
        draw.text((x0 + 10, 78), f"Controller [{side}]", fill=(255, 255, 255), font=font_sm)
        draw.ellipse((x0 + 60, 110, x0 + 128, 178), outline=(200, 200, 200), width=2)
        draw.text((x0 + 82, 138), "Joy", fill=(200, 200, 200), font=font_sm)

        grip_y = 200
        grip_color = (46, 204, 113) if highlight_grip else (90, 90, 90)
        draw.rounded_rectangle((x0 + 20, grip_y, x0 + 168, grip_y + 56), radius=8, fill=grip_color)
        draw.text((x0 + 30, grip_y + 16), f"{'LG' if side=='L' else 'RG'} GRIP", fill=(255, 255, 255), font=font)
        if highlight_grip:
            draw.text((x0 + 30, grip_y + 34), "HOLD (IK heart)", fill=(255, 255, 255), font=font_sm)

        draw.text((x0 + 20, 280), "LT / RT trigger", fill=(160, 160, 160), font=font_sm)
        draw.text((x0 + 20, 310), "A/B/X/Y buttons", fill=(160, 160, 160), font=font_sm)

    return img


def _compose(base: Image.Image, panel: Image.Image, banner: str) -> Image.Image:
    out = base.copy()
    out.paste(panel, (20, 20), panel)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, 1280, 42), fill=(0, 0, 0, 180))
    draw.text((450, 10), banner, fill=(255, 255, 255), font=font)
    return out


def upload_gofile() -> str | None:
    import requests

    with VIDEO_PATH.open("rb") as f:
        resp = requests.post(
            "https://upload.gofile.io/uploadFile",
            files={"file": (VIDEO_PATH.name, f, "video/mp4")},
            timeout=180,
        )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        return None
    return data["data"]["downloadPage"]


def main() -> int:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("frame_*.png"):
        old.unlink()

    print("Computing heart gesture IK trajectory...")
    robot = OpenArmRobot()
    controller = IKController(robot, PyrokiSolver(robot))
    urdf = yourdfpy.URDF.load(robot.urdf_path)
    joint_names = robot.actuated_joint_names
    left_chain = _resolve_chain(urdf, _LEFT_CHAIN)
    right_chain = _resolve_chain(urdf, _RIGHT_CHAIN)

    configs = _interpolate_heart_configs(robot, controller)
    frames: list[Image.Image] = []

    urdf.update_cfg({name: float(val) for name, val in zip(joint_names, configs[0])})
    intro = _render_robot_frame(
        urdf,
        left_chain,
        right_chain,
        "OpenArm — Chrome VR IK Demo",
        "TeleopXR + IWER virtual Quest 3 controllers",
    )
    intro = _compose(intro, _render_iwer_panel(False), "Step 1: Enter VR Mode in Chrome")
    frames.extend([intro] * 8)

    vr_panel = _render_iwer_panel(False)
    vr = _render_robot_frame(
        urdf,
        left_chain,
        right_chain,
        "VR Mode — immersive-vr session",
        "OpenArm model loaded in WebXR scene",
    )
    vr = _compose(vr, vr_panel, "Step 2: VR Mode active")
    frames.extend([vr] * 8)

    grip_panel = _render_iwer_panel(True)
    for i, q in enumerate(configs):
        cfg = {name: float(val) for name, val in zip(joint_names, q)}
        urdf.update_cfg(cfg)
        phase = "Step 3: Both grips held → Heart IK" if i > 0 else "Step 3: Press/hold LG + RG grip"
        subtitle = f"IK solving bimanual heart pose ({i}/{len(configs)-1})"
        img = _render_robot_frame(urdf, left_chain, right_chain, "OpenArm Heart Gesture", subtitle)
        img = _compose(img, grip_panel, phase)
        frames.append(img)

    for i, png in enumerate(frames):
        png.save(FRAME_DIR / f"frame_{i:04d}.png")

    print(f"Encoding {VIDEO_PATH} ({len(frames)} frames @ {FPS} fps)")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(FRAME_DIR / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(VIDEO_PATH),
        ],
        check=True,
    )

    link = upload_gofile()
    print(f"GOFILE_URL={link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
