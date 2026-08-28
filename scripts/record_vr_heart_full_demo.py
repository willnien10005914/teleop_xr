#!/usr/bin/env python3
"""Full demo: Chrome VR entry (Playwright) + OpenArm mesh heart animation → gofile."""

from __future__ import annotations

import io
import os
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import numpy as np
import yourdfpy
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

from teleop_xr.ik.controller import IKController
from teleop_xr.ik.gestures.heart import get_heart_pose_targets
from teleop_xr.ik.robots.openarm import OpenArmRobot
from teleop_xr.ik.solver import PyrokiSolver

HOST = os.environ.get("TELEOP_HOST", "127.0.0.1")
PORT = int(os.environ.get("TELEOP_PORT", "4443"))
URL = f"https://{HOST}:{PORT}/"
OUT_DIR = Path(os.environ.get("DEMO_OUT_DIR", "/home/testpc-3/projects/xr/results"))
FRAME_DIR = OUT_DIR / "vr_heart_frames"
VIDEO_PATH = OUT_DIR / "openarm_vr_heart_demo.mp4"
CACHE_PATH = OUT_DIR / "heart_configs.npy"
FPS = 10
IK_STEPS = 24


def wait_for_server(timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    ctx = ssl._create_unverified_context()
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL, context=ctx, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"Server not ready at {URL}")


def _fonts():
    try:
        return (
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13),
        )
    except OSError:
        d = ImageFont.load_default()
        return d, d, d


def _banner(img: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font_t, font_s, _ = _fonts()
    draw.rectangle((0, 0, 1280, 52 if subtitle else 44), fill=(0, 0, 0, 210))
    draw.text((24, 8), title, fill=(255, 255, 255), font=font_t)
    if subtitle:
        draw.text((24, 32), subtitle, fill=(200, 200, 200), font=font_s)
    return out


def _compute_heart_configs(robot: OpenArmRobot, controller: IKController) -> list[np.ndarray]:
    if CACHE_PATH.exists():
        arr = np.load(CACHE_PATH)
        return [arr[i] for i in range(len(arr))]

    q = np.array(robot.get_default_config())
    controller.set_mode("ee_absolute")
    heart = get_heart_pose_targets()
    fk0 = robot.forward_kinematics(jnp.asarray(q))
    configs = [q.copy()]

    for step in range(1, IK_STEPS + 1):
        alpha = step / IK_STEPS
        blended = {}
        for frame, target in heart.items():
            if frame not in fk0:
                continue
            sp = np.asarray(fk0[frame].translation(), dtype=float)
            sw = np.asarray(fk0[frame].rotation().wxyz, dtype=float)
            gp = np.array([target["position"][k] for k in ("x", "y", "z")], dtype=float)
            gw = np.array(
                [target["orientation"][k] for k in ("w", "x", "y", "z")], dtype=float
            )
            pos = sp + alpha * (gp - sp)
            wxyz = sw + alpha * (gw - sw)
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
        print(f"  IK step {step}/{IK_STEPS}")

    controller.set_mode("teleop")
    np.save(CACHE_PATH, np.stack(configs))
    return configs


def _render_mesh(urdf: yourdfpy.URDF, cfg: dict[str, float]) -> Image.Image:
    urdf.update_cfg(cfg)
    scene = urdf.scene
    center = scene.centroid
    scene.set_camera(
        distance=2.1,
        center=center,
        angles=(np.radians(22), 0.0, 0.0),
    )
    png = scene.save_image(resolution=[1280, 720], background=[18, 18, 28, 255])
    return Image.open(io.BytesIO(png)).convert("RGB")


def _capture_browser_entry() -> list[tuple[Image.Image, str, str]]:
    """Real Chrome UI: dashboard → VR Mode → IWER grips."""
    display = os.environ.get("DISPLAY", ":0")
    headed = os.environ.get("HEADED", "1") != "0"
    shots: list[tuple[Image.Image, str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=["--ignore-certificate-errors", "--enable-webgl", "--use-gl=angle"],
            env={**os.environ, "DISPLAY": display} if headed else None,
        )
        page = browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        ).new_page()

        page.goto(URL, wait_until="networkidle", timeout=120_000)
        page.wait_for_function("() => window.__iwer?.device", timeout=30_000)

        shots.append((
            Image.open(io.BytesIO(page.screenshot())).convert("RGB"),
            "TeleopXR — Open https://host:4443",
            "Step 1: Open dashboard in Chrome (HTTPS)",
        ))

        page.get_by_role("button", name="VR Mode").click()
        page.wait_for_timeout(1500)
        page.evaluate("() => window.__iwer?.grantSession?.()")
        page.wait_for_timeout(2500)

        shots.append((
            Image.open(io.BytesIO(page.screenshot())).convert("RGB"),
            "Enter VR Mode + IWER Meta Quest 3",
            "Step 2: Click VR Mode → immersive-vr session",
        ))

        # Hide 2D dashboard; keep IWER DevUI for VR view.
        page.add_style_tag(
            content="main > .relative.z-10 { visibility: hidden !important; }"
        )
        page.wait_for_timeout(3000)

        for grip in (0.3, 0.6, 1.0):
            page.evaluate(f"() => window.__iwer?.setGrip({grip}, {grip})")
            page.wait_for_timeout(400)
            shots.append((
                Image.open(io.BytesIO(page.screenshot())).convert("RGB"),
                f"Virtual grip squeeze {int(grip*100)}%",
                "Step 3: Hold LG + RG on IWER controllers",
            ))

        page.wait_for_timeout(800)
        shots.append((
            Image.open(io.BytesIO(page.screenshot())).convert("RGB"),
            "Both grips held — triggering heart IK",
            "Step 4: OpenArm IK heart gesture starts",
        ))
        browser.close()

    return shots


def upload_gofile() -> str:
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
        raise RuntimeError(f"gofile upload failed: {data}")
    return data["data"]["downloadPage"]


def main() -> int:
    wait_for_server()
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("frame_*.png"):
        old.unlink()

    print("Capturing Chrome VR entry (Playwright)...")
    browser_shots = _capture_browser_entry()

    print("Computing / loading heart IK trajectory...")
    robot = OpenArmRobot()
    controller = IKController(robot, PyrokiSolver(robot))
    urdf = yourdfpy.URDF.load(robot.urdf_path)
    joint_names = robot.actuated_joint_names
    configs = _compute_heart_configs(robot, controller)
    hold = [configs[-1]] * 15

    frames: list[Image.Image] = []

    # Repeat browser key frames for readable pacing.
    for img, title, sub in browser_shots[:2]:
        frames.extend([_banner(img, title, sub)] * 12)

    for img, title, sub in browser_shots[2:]:
        frames.extend([_banner(img, title, sub)] * 8)

    print("Rendering OpenArm mesh animation...")
    total = len(configs) + len(hold)
    for i, q in enumerate(configs + hold):
        cfg = {n: float(v) for n, v in zip(joint_names, q)}
        mesh = _render_mesh(urdf, cfg)
        phase = min(i, len(configs) - 1)
        sub = (
            f"OpenArm bimanual IK heart pose — frame {phase}/{len(configs)-1}"
            if i < len(configs)
            else "Heart pose hold"
        )
        frames.append(_banner(mesh, "OpenArm Heart Gesture (IK)", sub))

    for i, frame in enumerate(frames):
        frame.save(FRAME_DIR / f"frame_{i:04d}.png")

    print(f"Encoding {len(frames)} frames @ {FPS} fps → {VIDEO_PATH}")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(FRAME_DIR / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(VIDEO_PATH),
        ],
        check=True,
    )

    link = upload_gofile()
    print(f"GOFILE_URL={link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
