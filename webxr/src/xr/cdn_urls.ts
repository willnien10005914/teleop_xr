/** Same-origin copies of CDN assets (see scripts/vendor-xr-assets.mjs). */
export const LOCAL_PROFILES_PATH = "/webxr-input-profiles";
export const LOCAL_DRACO_PATH = "/draco/gltf/";
export const LOCAL_BASIS_PATH = "/basis/";

const CDN_PROFILES_RE =
	/^https:\/\/cdn\.jsdelivr\.net\/npm\/@webxr-input-profiles\/assets@[^/]+\/dist\/profiles(\/.*)?$/;
const UNPKG_THREE_LIBS_RE =
	/^https:\/\/unpkg\.com\/three@[^/]+\/examples\/jsm\/libs\/(draco\/gltf|basis)(\/.*)?$/;

/**
 * Map firewall-blocked CDN URLs (jsDelivr controller GLBs, unpkg Draco/Basis)
 * onto files served from this app's origin.
 *
 * Samsung Galaxy XR controller GLBs are ~5MB each and can OOM Chrome after
 * the session starts. Use the smaller generic thumbstick meshes instead.
 */
const LIGHT_CONTROLLER_PROFILE = "generic-trigger-squeeze-thumbstick";
const HEAVY_CONTROLLER_PROFILES = ["samsung-galaxyxr"];

export function rewriteCdnAssetUrl(url: string): string {
	let rewritten = url;
	const profiles = rewritten.match(CDN_PROFILES_RE);
	if (profiles) {
		rewritten = `${LOCAL_PROFILES_PATH}${profiles[1] ?? ""}`;
	}
	const threeLib = rewritten.match(UNPKG_THREE_LIBS_RE);
	if (threeLib) {
		rewritten = `/${threeLib[1]}${threeLib[2] ?? "/"}`;
	}
	for (const heavy of HEAVY_CONTROLLER_PROFILES) {
		rewritten = rewritten.replaceAll(
			`/${heavy}/`,
			`/${LIGHT_CONTROLLER_PROFILE}/`,
		);
	}
	return rewritten;
}
