#!/usr/bin/env python3
"""Record OpenArm VR heart-gesture demo (Chrome + IWER) and upload to gofile."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import ssl
from pathlib import Path

from playwright.sync_api import sync_playwright

HOST = os.environ.get("TELEOP_HOST", "127.0.0.1")
PORT = int(os.environ.get("TELEOP_PORT", "4443"))
OUT_DIR = Path(os.environ.get("DEMO_OUT_DIR", "/home/testpc-3/projects/xr/results"))
FRAME_DIR = OUT_DIR / "vr_heart_frames"
VIDEO_PATH = OUT_DIR / "openarm_vr_heart_demo.mp4"
FPS = 4
URL = f"https://{HOST}:{PORT}/"


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


def upload_gofile() -> str | None:
    if not VIDEO_PATH.exists():
        return None
    try:
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
    except Exception as exc:
        print(f"gofile upload failed: {exc}", file=sys.stderr)
        return None


def capture_with_playwright() -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("frame_*.png"):
        old.unlink()

    display = os.environ.get("DISPLAY", ":0")
    headed = os.environ.get("HEADED", "1") != "0"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=[
                "--ignore-certificate-errors",
                "--enable-webgl",
                "--use-gl=angle",
                "--enable-features=WebXR",
            ],
            env={**os.environ, "DISPLAY": display} if headed else None,
        )
        page = browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        ).new_page()

        frames: list[bytes] = []

        def grab(label: str = "") -> None:
            shot: bytes | None = None
            best_area = 0.0
            for canvas in page.locator("canvas").all():
                try:
                    box = canvas.bounding_box()
                    if not box:
                        continue
                    area = box["width"] * box["height"]
                    if area > best_area:
                        best_area = area
                        shot = canvas.screenshot()
                except Exception:
                    continue
            if shot is None:
                shot = page.screenshot(full_page=False)
            frames.append(shot)
            if label:
                print(f"  frame {len(frames):02d}: {label}")

        page.goto(URL, wait_until="networkidle", timeout=120_000)
        page.wait_for_function("() => window.__iwer?.device", timeout=30_000)
        grab("IWER ready")

        page.get_by_role("button", name="VR Mode").click()
        page.wait_for_timeout(2000)

        # IWER requires granting the offered immersive session.
        enter_xr = page.get_by_role("button", name="Enter XR")
        if enter_xr.count():
            enter_xr.click()
            print("  clicked IWER Enter XR")
        page.wait_for_timeout(4000)

        # Hide 2D dashboard overlay; keep IWER DevUI visible.
        page.add_style_tag(
            content="main > .relative.z-10 { visibility: hidden !important; }"
        )
        page.wait_for_timeout(8000)
        grab("VR session started")

        # Both grips at 100% via IWER API (reliable vs DevUI slider DOM).
        page.evaluate(
            """() => {
                window.__iwer?.setGrip(1, 1);
            }"""
        )
        grab("grips ON")

        for i in range(14):
            page.wait_for_timeout(300)
            grab(f"heart hold {i}")

        page.evaluate("() => { window.__iwer?.setStick(-0.8, -0.8); }")
        for i in range(8):
            page.wait_for_timeout(300)
            grab(f"joystick {i}")

        page.evaluate(
            """() => {
                window.__iwer?.setGrip(0, 0);
                window.__iwer?.setStick(0, 0);
            }"""
        )
        for i in range(8):
            page.wait_for_timeout(400)
            grab(f"release {i}")

        for i, png in enumerate(frames):
            (FRAME_DIR / f"frame_{i:04d}.png").write_bytes(png)

        browser.close()


def main() -> int:
    print(f"Waiting for demo server at {URL}")
    wait_for_server()
    print("Recording VR heart demo...")
    capture_with_playwright()
    print(f"Encoding {VIDEO_PATH}")
    encode_video()
    link = upload_gofile()
    if link:
        print(f"GOFILE_URL={link}")
    else:
        print(f"Video saved locally: {VIDEO_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
