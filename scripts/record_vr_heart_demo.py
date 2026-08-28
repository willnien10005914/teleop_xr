#!/usr/bin/env python3
"""Capture OpenArm VR heart-gesture demo with Playwright screenshots + WebSocket IK drive."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import subprocess
import sys
import time
from pathlib import Path

import websockets
from playwright.sync_api import sync_playwright

HOST = os.environ.get("TELEOP_HOST", "127.0.0.1")
PORT = int(os.environ.get("TELEOP_PORT", "4443"))
OUT_DIR = Path(os.environ.get("DEMO_OUT_DIR", "/home/testpc-3/projects/xr/results"))
FRAME_DIR = OUT_DIR / "vr_heart_frames"
VIDEO_PATH = OUT_DIR / "openarm_vr_heart_demo.mp4"
DURATION_S = float(os.environ.get("DEMO_DURATION_S", "24"))
FPS = 2
URL = f"https://{HOST}:{PORT}/"


def _pose(x: float, y: float, z: float) -> dict:
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _gamepad(squeeze: bool, stick_y: float = 0.0) -> dict:
    return {
        "buttons": [
            {"pressed": False, "touched": False, "value": 0.0},
            {"pressed": squeeze, "touched": squeeze, "value": 1.0 if squeeze else 0.0},
            {"pressed": False, "touched": False, "value": 0.0},
            {"pressed": False, "touched": False, "value": 0.0},
        ],
        "axes": [0.0, stick_y, 0.0, stick_y],
    }


def _xr_state(t_ms: float, left_squeeze: bool, right_squeeze: bool, stick_y: float = 0.0) -> dict:
    return {
        "timestamp_unix_ms": t_ms,
        "devices": [
            {"role": "head", "handedness": "none", "pose": _pose(0.0, 1.6, 0.0)},
            {
                "role": "controller",
                "handedness": "left",
                "gripPose": _pose(-0.2, 1.2, -0.3),
                "gamepad": _gamepad(left_squeeze, stick_y),
            },
            {
                "role": "controller",
                "handedness": "right",
                "gripPose": _pose(0.2, 1.2, -0.3),
                "gamepad": _gamepad(right_squeeze, stick_y),
            },
        ],
        "fps": 60,
        "fetch_latency_ms": 1.0,
    }


async def drive_xr_session(stop_event: asyncio.Event) -> None:
    uri = f"wss://{HOST}:{PORT}/ws"
    ssl_ctx = ssl._create_unverified_context()
    async with websockets.connect(uri, ssl=ssl_ctx) as ws:
        for _ in range(3):
            try:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
            except TimeoutError:
                break

        t0 = time.time()
        while not stop_event.is_set() and time.time() - t0 < DURATION_S:
            elapsed = time.time() - t0
            t_ms = time.time() * 1000.0
            left = right = False
            stick_y = 0.0
            if 2.0 <= elapsed < 14.0:
                left = right = True
            if 6.0 <= elapsed < 9.0:
                stick_y = 0.85
            payload = {
                "type": "xr_state",
                "client_id": "vr-heart-recorder",
                "data": _xr_state(t_ms, left, right, stick_y),
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(0.05)


def wait_for_server(timeout_s: float = 120.0) -> None:
    import urllib.request

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


def encode_video() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        "-vf",
        "scale=1280:720",
        str(VIDEO_PATH),
    ]
    subprocess.run(cmd, check=True)


def upload_video() -> str | None:
    if not VIDEO_PATH.exists():
        return None
    try:
        import requests

        with VIDEO_PATH.open("rb") as f:
            resp = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": (VIDEO_PATH.name, f, "video/mp4")},
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        url = data.get("data", {}).get("url", "")
        if url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        if "tmpfiles.org/" in url and "/dl/" not in url:
            url = url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)
        return url
    except Exception as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return None


def capture_with_playwright() -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("frame_*.png"):
        old.unlink()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--enable-webxr",
                "--enable-features=WebXR",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )
        page = context.new_page()
        page.goto(URL, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(2000)

        # Enter VR mode (IWER polyfill provides WebXR on desktop Chrome).
        vr_btn = page.get_by_role("button", name="VR Mode")
        if vr_btn.count():
            vr_btn.click()
            page.wait_for_timeout(8000)

        # Hold LG + RG via IWER DevUI "Hold" toggles (2nd Hold per controller panel).
        for panel_label in ("Controller [L]", "Controller [R]"):
            panel = page.locator(f"text={panel_label}").locator("xpath=ancestor::div[1]")
            grip_hold = panel.locator("text=Hold").nth(1)
            if grip_hold.count():
                grip_hold.click()
        page.wait_for_timeout(1500)

        # Nudge virtual thumbsticks forward during grip hold.
        page.keyboard.down("KeyW")
        page.keyboard.down("ArrowUp")
        page.wait_for_timeout(1500)
        page.keyboard.up("KeyW")
        page.keyboard.up("ArrowUp")
        page.wait_for_timeout(2000)

        for panel_label in ("Controller [L]", "Controller [R]"):
            panel = page.locator(f"text={panel_label}").locator("xpath=ancestor::div[1]")
            grip_hold = panel.locator("text=Hold").nth(1)
            if grip_hold.count():
                grip_hold.click()
        page.wait_for_timeout(2000)

        total_frames = int(DURATION_S * FPS)
        for i in range(total_frames):
            page.screenshot(path=str(FRAME_DIR / f"frame_{i:04d}.png"), full_page=False)
            time.sleep(1.0 / FPS)

        browser.close()


def main() -> int:
    print(f"Waiting for demo server at {URL}")
    wait_for_server()
    print("Capturing Playwright screenshots while driving XR grips/joystick...")
    capture_with_playwright()
    print(f"Encoding {VIDEO_PATH}")
    encode_video()
    link = upload_video()
    if link:
        print(f"UPLOAD_URL={link}")
    else:
        print(f"Video saved locally: {VIDEO_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
