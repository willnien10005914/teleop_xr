import { describe, expect, it } from "vitest";
import { rewriteCdnAssetUrl } from "../cdn_urls";

describe("rewriteCdnAssetUrl", () => {
	it("maps Samsung Galaxy XR controller meshes onto the smaller generic profile", () => {
		expect(
			rewriteCdnAssetUrl(
				"https://cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0/dist/profiles/samsung-galaxyxr/left.glb",
			),
		).toBe("/webxr-input-profiles/generic-trigger-squeeze-thumbstick/left.glb");
	});

	it("maps unpkg Draco decoder files onto the local public path", () => {
		expect(
			rewriteCdnAssetUrl(
				"https://unpkg.com/three@0.177.0/examples/jsm/libs/draco/gltf/draco_decoder.wasm",
			),
		).toBe("/draco/gltf/draco_decoder.wasm");
	});

	it("leaves same-origin URLs unchanged", () => {
		expect(rewriteCdnAssetUrl("/robot_assets/openarm/link.glb")).toBe(
			"/robot_assets/openarm/link.glb",
		);
	});
});
