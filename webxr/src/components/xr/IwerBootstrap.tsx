"use client";

import { useEffect } from "react";

/**
 * Installs Meta IWER (Immersive Web Emulation Runtime) so Chrome desktop
 * can enter WebXR VR mode with virtual Quest controllers + thumbsticks.
 */
export function IwerBootstrap() {
	useEffect(() => {
		let cancelled = false;

		const install = async () => {
			if (typeof window === "undefined" || cancelled) return;
			if (!window.isSecureContext) {
				console.warn("[IwerBootstrap] WebXR requires a secure context (HTTPS).");
				return;
			}
			if (navigator.xr && (navigator.xr as XRSystem & { isPolyfill?: boolean }).isPolyfill) {
				return;
			}

			try {
				const [{ XRDevice, metaQuest3 }, { DevUI }] = await Promise.all([
					import("iwer"),
					import("@iwer/devui"),
				]);
				if (cancelled) return;

				const device = new XRDevice(metaQuest3);
				device.installRuntime();
				device.installDevUI(DevUI);
				console.info("[IwerBootstrap] IWER installed (Meta Quest 3 emulation)");
			} catch (err) {
				console.error("[IwerBootstrap] Failed to install IWER:", err);
			}
		};

		void install();

		return () => {
			cancelled = true;
		};
	}, []);

	return null;
}
