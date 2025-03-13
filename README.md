# KO Drum Design Optimization

This project implements a Knock-Out (KO) Drum design optimization tool for gas-liquid separation in process engineering applications. The tool analyzes and optimizes various design parameters to achieve maximum water recovery efficiency.

## Features

- Base and optimized KO drum configurations
- Mass and energy balance analysis
- Water recovery performance evaluation
- Visualization of key performance metrics
- Parameter sensitivity analysis
- Stream composition analysis

## Key Components

- `ko_drum_physics.py`: Core physics simulation for the KO drum
- `optimised_drum_config.py`: Optimized configuration parameters
- `analyze_mass_energy_balance.py`: Analysis and visualization tools

## Results

The optimized design achieves:
- Improved water recovery (98.97% vs 93.38%)
- Optimized L/D ratio of 5.02
- Enhanced natural separation through temperature optimization
- Efficient mechanical separation

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- CoolProp

## Usage

```bash
# Run the analysis
python analyze_mass_energy_balance.py
```

## License

MIT License 