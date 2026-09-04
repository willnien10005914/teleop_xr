import path from "node:path";
import fs from "fs-extra";

/**
 * Copy WebXR controller GLBs and Draco/Basis decoders into public/ so the
 * headset never has to reach jsDelivr or unpkg (often blocked by firewalls).
 */
export async function vendorXrRuntimeAssets() {
	const root = process.cwd();
	const profilesSrc = path.join(
		root,
		"node_modules/@webxr-input-profiles/assets/dist/profiles",
	);
	const profilesDest = path.join(root, "public/webxr-input-profiles");
	if (!(await fs.pathExists(profilesSrc))) {
		throw new Error(
			"Missing @webxr-input-profiles/assets. Run npm install in webxr/.",
		);
	}

	await fs.emptyDir(profilesDest);
	await fs.copy(profilesSrc, profilesDest);
	console.log(
		`✅ Vendored WebXR input profiles -> ${path.relative(root, profilesDest)}`,
	);

	const threeLibs = path.join(root, "node_modules/three/examples/jsm/libs");
	const dracoSrc = path.join(threeLibs, "draco/gltf");
	const basisSrc = path.join(threeLibs, "basis");
	if (!(await fs.pathExists(dracoSrc)) || !(await fs.pathExists(basisSrc))) {
		throw new Error("Missing three.js Draco/Basis libs under node_modules/three.");
	}

	const dracoDest = path.join(root, "public/draco/gltf");
	const basisDest = path.join(root, "public/basis");
	await fs.emptyDir(dracoDest);
	await fs.emptyDir(basisDest);
	await fs.copy(dracoSrc, dracoDest);
	await fs.copy(basisSrc, basisDest);
	console.log(
		`✅ Vendored Draco/Basis decoders -> public/draco/gltf, public/basis`,
	);
}
