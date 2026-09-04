import { describe, expect, it, vi } from "vitest";
import {
	buildXrSessionInit,
	IMMERSIVE_AR,
	IMMERSIVE_VR,
	preferredSessionMode,
	requestPreferredXrSession,
} from "../session";

describe("preferredSessionMode", () => {
	it("uses immersive-ar for VR so Samsung XR does not black out", () => {
		expect(preferredSessionMode("vr")).toBe(IMMERSIVE_AR);
		expect(preferredSessionMode("passthrough")).toBe(IMMERSIVE_AR);
	});

	it("omits anchors, layers, and dom-overlay from the session init", () => {
		expect(buildXrSessionInit()).toEqual({
			optionalFeatures: ["local-floor", "hand-tracking"],
		});
	});
});

describe("requestPreferredXrSession", () => {
	it("requests immersive-ar when it is supported", async () => {
		const xr = {
			isSessionSupported: vi.fn().mockResolvedValue(true),
			requestSession: vi.fn().mockResolvedValue({ id: "ar" }),
		};
		await requestPreferredXrSession(xr, "vr", {});
		expect(xr.requestSession).toHaveBeenCalledWith(IMMERSIVE_AR, {});
	});

	it("falls back to immersive-vr when AR is not supported", async () => {
		const xr = {
			isSessionSupported: vi.fn().mockResolvedValue(false),
			requestSession: vi.fn().mockResolvedValue({ id: "vr" }),
		};
		await requestPreferredXrSession(xr, "vr", { optionalFeatures: [] });
		expect(xr.requestSession).toHaveBeenCalledWith(IMMERSIVE_VR, {
			optionalFeatures: [],
		});
	});
});
