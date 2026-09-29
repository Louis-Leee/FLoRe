import type * as THREE from 'three';

declare module 'three' {
  interface Object3D {
    bodyID?: number | string;
    has_custom_mesh?: boolean;
    cylinders?: THREE.InstancedMesh;
    spheres?: THREE.InstancedMesh;
  }

  interface Mesh {
    bodyID?: number | string;
    has_custom_mesh?: boolean;
  }

  interface Group {
    bodyID?: number | string;
    has_custom_mesh?: boolean;
    cylinders?: THREE.InstancedMesh;
    spheres?: THREE.InstancedMesh;
  }
}
