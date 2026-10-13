import { Group, Mesh, MeshStandardMaterial, type BufferGeometry, type Object3D } from 'three';
import { modelFromObject, type Model } from './model.ts';

export const SUPPORTED_EXTENSIONS = ['glb', 'gltf', 'obj', 'stl', 'ply', 'fbx'];

function wrapGeometry(geometry: BufferGeometry): Object3D {
  const group = new Group();
  group.add(new Mesh(geometry, new MeshStandardMaterial({ color: 0xb8bcc4 })));
  return group;
}

/** Reads a user-supplied 3D file into a normalised triangle model. Loaders are fetched on demand. */
export async function loadModelFile(file: File): Promise<Model> {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  const name = file.name.replace(/\.[^.]+$/, '');
  let root: Object3D;

  switch (ext) {
    case 'glb':
    case 'gltf': {
      const [{ GLTFLoader }, { DRACOLoader }, { MeshoptDecoder }] = await Promise.all([
        import('three/examples/jsm/loaders/GLTFLoader.js'),
        import('three/examples/jsm/loaders/DRACOLoader.js'),
        import('three/examples/jsm/libs/meshopt_decoder.module.js'),
      ]);
      const loader = new GLTFLoader();
      const draco = new DRACOLoader();
      draco.setDecoderPath(`${import.meta.env.BASE_URL}draco/`);
      loader.setDRACOLoader(draco);
      loader.setMeshoptDecoder(MeshoptDecoder);
      const buffer = await file.arrayBuffer();
      try {
        const gltf = await loader.parseAsync(buffer, '');
        root = gltf.scene;
      } catch (error) {
        const hint = ext === 'gltf' ? ' A .gltf file that references separate .bin or image files cannot be read on its own; export a single .glb instead.' : '';
        throw new Error(`Could not read the glTF file.${hint} (${describe(error)})`);
      } finally {
        draco.dispose();
      }
      break;
    }
    case 'obj': {
      const { OBJLoader } = await import('three/examples/jsm/loaders/OBJLoader.js');
      root = new OBJLoader().parse(await file.text());
      break;
    }
    case 'stl': {
      const { STLLoader } = await import('three/examples/jsm/loaders/STLLoader.js');
      root = wrapGeometry(new STLLoader().parse(await file.arrayBuffer()));
      break;
    }
    case 'ply': {
      const { PLYLoader } = await import('three/examples/jsm/loaders/PLYLoader.js');
      root = wrapGeometry(new PLYLoader().parse(await file.arrayBuffer()));
      break;
    }
    case 'fbx': {
      const { FBXLoader } = await import('three/examples/jsm/loaders/FBXLoader.js');
      root = new FBXLoader().parse(await file.arrayBuffer(), '');
      break;
    }
    default:
      throw new Error(`Unsupported file type ".${ext}". Use ${SUPPORTED_EXTENSIONS.map((e) => `.${e}`).join(', ')}.`);
  }
  return modelFromObject(root, name);
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
