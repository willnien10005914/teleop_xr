import numpy as np
import pytest

from teleop_xr.ik.gestures.triggers import BothGripHeartTrigger
from teleop_xr.messages import (
    XRButtonState,
    XRDeviceRole,
    XRGamepad,
    XRHandedness,
    XRInputSource,
    XRState,
)


def _squeeze_gamepad(pressed: bool) -> XRGamepad:
    return XRGamepad(
        buttons=[
            XRButtonState(pressed=False, touched=False, value=0.0),
            XRButtonState(pressed=pressed, touched=pressed, value=1.0 if pressed else 0.0),
        ],
        axes=[0.0, 0.0],
    )


def _state(t_ms: float, left: bool, right: bool) -> XRState:
    return XRState(
        timestamp_unix_ms=t_ms,
        devices=[
            XRInputSource(
                role=XRDeviceRole.CONTROLLER,
                handedness=XRHandedness.LEFT,
                gamepad=_squeeze_gamepad(left),
            ),
            XRInputSource(
                role=XRDeviceRole.CONTROLLER,
                handedness=XRHandedness.RIGHT,
                gamepad=_squeeze_gamepad(right),
            ),
        ],
    )


def test_both_grip_heart_trigger_fires_once_per_hold():
    trigger = BothGripHeartTrigger(threshold_ms=800.0)

    assert trigger.update(_state(0.0, True, True)) is False
    assert trigger.update(_state(500.0, True, True)) is False
    assert trigger.update(_state(900.0, True, True)) is True
    assert trigger.update(_state(1000.0, True, True)) is False

    assert trigger.update(_state(1100.0, False, True)) is False
    assert trigger.update(_state(1200.0, True, True)) is False
    assert trigger.update(_state(2100.0, True, True)) is True


def test_both_grip_heart_trigger_reset():
    trigger = BothGripHeartTrigger(threshold_ms=100.0)
    trigger.update(_state(0.0, True, True))
    trigger.update(_state(200.0, True, True))
    trigger.reset()
    assert trigger.update(_state(250.0, True, True)) is False
    assert trigger.update(_state(400.0, True, True)) is True
