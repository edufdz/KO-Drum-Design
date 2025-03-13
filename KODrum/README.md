# OpenFOAM Knockout Drum Simulation

This case simulates a gas-liquid separation process in a Knockout Drum using OpenFOAM's interFoam solver.

## Geometry Specifications
- Length: 4.0 m
- Diameter: 1.45 m
- Mist eliminator location: 3.0 m from inlet

## Operating Conditions
- Temperature: 50°C
- Pressure: 1 atm
- Gas flowrate: 5.3 m³/s
- Liquid fraction: 10% at inlet
- Target droplet size: 10μm

## Running the Simulation

1. Generate the mesh:
```bash
blockMesh
```

2. Initialize fields:
```bash
setFields
```

3. Run the simulation:
```bash
interFoam > log.interFoam 2>&1
```

4. Monitor progress:
```bash
tail -f log.interFoam
```

5. Post-process results:
```bash
paraFoam
```

## Expected Results
- Gas-phase purity: >99% at outlet
- Droplet removal efficiency: >99% for droplets ≥10μm
- Maximum gas velocity: ≤4 m/s (Souders-Brown limit)

## Files Description
- `system/blockMeshDict`: Mesh generation settings
- `system/controlDict`: Simulation control parameters
- `system/fvSchemes`: Numerical schemes
- `system/fvSolution`: Solver settings
- `constant/transportProperties`: Fluid properties
- `0/`: Initial conditions

## Post-processing
Use ParaView to visualize:
- Phase distribution
- Velocity profiles
- Pressure fields
- Droplet trajectories 