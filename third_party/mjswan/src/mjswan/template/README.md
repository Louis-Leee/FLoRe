# mjswan Web Viewer

Browser-based interactive viewer for MuJoCo robotics simulations with real-time policy control.

## Overview

A React + TypeScript application that runs MuJoCo physics simulations entirely in the browser using WebAssembly, featuring 3D visualization with Three.js and real-time ONNX policy execution.

## Key Features

- MuJoCo physics simulation via WebAssembly
- Interactive 3D rendering with Three.js
- Real-time ONNX policy execution
- Drag-to-apply forces on objects
- Multi-project and multi-scene support
- WebXR/VR ready
- Static site output for easy deployment

## Technology Stack

- React 18 + TypeScript
- Three.js (WebGL 2.0)
- MuJoCo WebAssembly ([mujoco-js](https://github.com/google-deepmind/mujoco/tree/main/javascript))
- ONNX Runtime Web
- Mantine UI
- Vite

## Installation

```bash
npm install mjswan
```

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Type check
npm run typecheck

# Lint
npm run lint
```

## Configuration

The viewer requires a configuration file at `assets/config.json`:

```json
{
  "version": "0.0.4",
  "projects": [
    {
      "name": "Project Name",
      "id": "project_id",
      "scenes": [
        {
          "name": "Scene Name",
          "path": "scene/scene_name/scene.xml",
          "policies": [
            {
              "name": "Policy Name",
              "source": "policy/scene_name/policy.onnx"
            }
          ]
        }
      ]
    }
  ]
}
```

### Asset Structure

```
assets/
├── config.json
├── scene/
│   └── {scene_name}/
│       └── scene.xml
└── policy/
    └── {scene_name}/
        └── {policy}.onnx
```

## URL Routing

- `/` - Default project
- `/{project-id}/` - Specific project
- `?scene={name}&policy={name}` - Pre-select scene and policy
- `?panel=0` - Start with the control panel hidden

## Core Architecture

### mjswanRuntime

Manages simulation, rendering, policy inference, and user interactions.
Located in [src/core/runtime.ts](src/core/runtime.ts)

### Scene Loading

Handles MJCF XML parsing, asset loading, and Three.js mesh generation.
Located in [src/core/scene.ts](src/core/scene.ts)

### Policy Execution

1. Extract observations from MuJoCo state
2. Run ONNX inference
3. Apply actions to simulation

## Project Structure

```
src/
├── App.tsx                  # Main application with routing
├── index.tsx                # Application entry point
├── index.css                # Global styles
├── components/              # React components
│   └── mjswanViewer.tsx    # Main viewer component
├── core/                    # Core engine
│   ├── engine/             # Physics simulation
│   ├── scene/              # Three.js scene setup
│   └── utils/              # Helper utilities
├── types/                   # TypeScript type definitions
└── utils/                   # Utility functions
```

### Base Path Configuration

For subdirectory deployment, update `vite.config.ts`:

```typescript
export default defineConfig({
  base: '/your-repo-name/',
})
```

## License

Apache-2.0

## Links

- **Repository**: [github.com/ttktjmt/mjswan](https://github.com/ttktjmt/mjswan)
- **Author**: Tatsuki Tsujimoto (tatsuki.tsujimoto@gmail.com)
