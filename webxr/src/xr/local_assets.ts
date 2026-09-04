import { AssetManager, type World } from "@iwsdk/core";
import {
	LOCAL_BASIS_PATH,
	LOCAL_DRACO_PATH,
	rewriteCdnAssetUrl,
} from "./cdn_urls";

export {
	LOCAL_BASIS_PATH,
	LOCAL_DRACO_PATH,
	LOCAL_PROFILES_PATH,
	rewriteCdnAssetUrl,
} from "./cdn_urls";

let loadGltfPatched = false;

/** Rewrite IWSDK controller-model fetches away from jsDelivr. */
export function patchAssetManagerForLocalCdn(): void {
	if (loadGltfPatched) {
		return;
	}
	loadGltfPatched = true;
	const original = AssetManager.loadGLTF.bind(AssetManager);
	AssetManager.loadGLTF = (url: string, key?: string) => {
		const rewritten = rewriteCdnAssetUrl(url);
		if (rewritten !== url) {
			console.info(`[XR assets] Loading controller/mesh from ${rewritten}`);
		}
		return original(rewritten, key);
	};
}

/** Point IWSDK Draco/KTX2 loaders at same-origin decoder binaries. */
export function configureLocalXrDecoders(world: World): void {
	AssetManager.init(world.renderer, world, {
		dracoDecoderPath: LOCAL_DRACO_PATH,
		ktx2TranscoderPath: LOCAL_BASIS_PATH,
	});
}
