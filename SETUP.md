# Setting up OpenFOAM for KO Drum Simulation

## Prerequisites Installation

1. Install Docker Desktop for Mac:
   - Download from: https://www.docker.com/products/docker-desktop
   - Follow the installation instructions
   - Start Docker Desktop after installation

2. Pull the OpenFOAM Docker image:
```bash
docker pull openfoam/openfoam10-paraview510
```

3. Create a Docker container for OpenFOAM:
```bash
docker run -it --name kofoam \
  -v $PWD:/home/openfoam/kodrumcase \
  openfoam/openfoam10-paraview510
```

## Running the Simulation

1. Start the OpenFOAM container (if not already running):
```bash
docker start kofoam
docker exec -it kofoam bash
```

2. Inside the container, navigate to the case directory:
```bash
cd $HOME/kodrumcase/KODrum
```

3. Generate the mesh:
```bash
blockMesh
```

4. Initialize fields:
```bash
setFields
```

5. Run the simulation:
```bash
interFoam > log.interFoam 2>&1
```

6. Monitor progress:
```bash
tail -f log.interFoam
```

## Visualization with ParaView

1. In a new terminal, start ParaView container:
```bash
docker run -it --rm \
  -v $PWD:/data \
  -e DISPLAY=host.docker.internal:0 \
  openfoam/openfoam10-paraview510 paraFoam
```

Note: For ParaView visualization on macOS, you'll need:
1. Install XQuartz (X11 server):
   - Download from: https://www.xquartz.org/
   - After installation, restart your computer
2. In XQuartz preferences, enable "Allow connections from network clients"
3. Run in terminal before starting ParaView:
   ```bash
   xhost + 127.0.0.1
   ```

## Troubleshooting

If you encounter permission issues:
```bash
chmod -R 777 KODrum/
```

If ParaView doesn't connect:
1. Ensure XQuartz is running
2. Check Docker permissions
3. Try restarting Docker Desktop

## Additional Notes

- The simulation files are mounted in the Docker container at `/home/openfoam/kodrumcase`
- All OpenFOAM commands should be run inside the Docker container
- Data is preserved in your local directory even when the container is stopped
- For large simulations, consider increasing Docker's resource allocation in Docker Desktop preferences 