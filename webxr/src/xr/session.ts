import type { XRMode } from "@/app/page";

type ActiveXrMode = Exclude<XRMode, null>;

export const IMMERSIVE_AR = "immersive-ar" as const;
export const IMMERSIVE_VR = "immersive-vr" as const;

/**
 * Keep the session feature set small. Anchors, layers, and dom-overlay have
 * crashed Samsung Galaxy XR Chrome after the scene appears for a second.
 */
export const XR_SESSION_OPTIONAL_FEATURES: string[] = [
	"local-floor",
	"hand-tracking",
];

export function buildXrSessionInit(): XRSessionInit {
	return {
		optionalFeatures: [...XR_SESSION_OPTIONAL_FEATURES],
	};
}

/**
 * Samsung Galaxy XR (and other Android XR browsers) often composite
 * `immersive-vr` as a black framebuffer. TeleopXR already draws a skybox for
 * "VR Mode", so prefer `immersive-ar` for both buttons. Fall back to VR only
 * when AR is not advertised (older headsets / some emulators).
 */
export function preferredSessionMode(mode: ActiveXrMode): XRSessionMode {
	void mode;
	return IMMERSIVE_AR;
}

export async function requestPreferredXrSession(
	xr: Pick<XRSystem, "isSessionSupported" | "requestSession">,
	mode: ActiveXrMode,
	sessionInit: XRSessionInit,
): Promise<XRSession> {
	const preferred = preferredSessionMode(mode);
	try {
		if (await xr.isSessionSupported(preferred)) {
			return xr.requestSession(preferred, sessionInit);
		}
	} catch {
		// Some browsers throw from isSessionSupported for unknown modes.
	}

	if (mode === "vr") {
		return xr.requestSession(IMMERSIVE_VR, sessionInit);
	}
	return xr.requestSession(preferred, sessionInit);
}
