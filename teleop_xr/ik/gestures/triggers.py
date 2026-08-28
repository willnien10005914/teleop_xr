"""Detect simultaneous both-grip hold to trigger the heart gesture."""

from __future__ import annotations

from dataclasses import dataclass, field

from teleop_xr.messages import (
    XRDeviceRole,
    XRHandedness,
    XRState,
)


@dataclass
class BothGripHeartTrigger:
    """Fire once when both controller grips stay pressed past the threshold."""

    threshold_ms: float = 800.0
    _both_pressed_since_ms: float | None = field(default=None, init=False)
    _fired_for_hold: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._both_pressed_since_ms = None
        self._fired_for_hold = False

    @staticmethod
    def _is_squeezed(state: XRState, handedness: XRHandedness) -> bool:
        for device in state.devices:
            if device.role != XRDeviceRole.CONTROLLER:
                continue
            if device.handedness != handedness:
                continue
            gamepad = device.gamepad
            if gamepad is None or len(gamepad.buttons) < 2:
                return False
            return bool(gamepad.buttons[1].pressed)
        return False

    def update(self, state: XRState) -> bool:
        """Return True once per both-grip hold when threshold is exceeded."""

        left = self._is_squeezed(state, XRHandedness.LEFT)
        right = self._is_squeezed(state, XRHandedness.RIGHT)
        now_ms = float(state.timestamp_unix_ms)

        if left and right:
            if self._both_pressed_since_ms is None:
                self._both_pressed_since_ms = now_ms
                self._fired_for_hold = False
            if (
                not self._fired_for_hold
                and now_ms - self._both_pressed_since_ms >= self.threshold_ms
            ):
                self._fired_for_hold = True
                return True
            return False

        self._both_pressed_since_ms = None
        self._fired_for_hold = False
        return False
