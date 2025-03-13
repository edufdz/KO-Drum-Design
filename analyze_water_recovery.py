import numpy as np
import matplotlib.pyplot as plt
from ko_drum_physics import KODrumPhysics

def main():
    # Create KO Drum simulator
    ko_drum = KODrumPhysics()
    
    # Temperature range from 20°C to 70°C
    temperatures = np.linspace(273.15 + 20, 273.15 + 70, 50)  # K
    
    print("Analyzing water recovery across temperature range...")
    print("Temperature range: 20°C to 70°C")
    print("Considering both vapor-liquid equilibrium and droplet separation efficiency")
    
    # Run analysis
    fig, results = ko_drum.plot_temperature_water_recovery(temperatures)
    
    # Print summary of results
    print("\nSummary of Results:")
    print("------------------")
    summary = results.describe()
    print("\nWater Recovery Statistics (%):")
    print(f"Mean Natural Liquid: {summary['Natural_Liquid_Fraction']['mean']*100:.1f}%")
    print(f"Mean Total Recovery: {summary['Total_Water_Recovery']['mean']*100:.1f}%")
    print(f"Min Total Recovery: {summary['Total_Water_Recovery']['min']*100:.1f}%")
    print(f"Max Total Recovery: {summary['Total_Water_Recovery']['max']*100:.1f}%")
    
    print("\nDroplet Separation Efficiency Statistics (%):")
    print(f"Mean Efficiency: {summary['Droplet_Separation_Efficiency']['mean']*100:.1f}%")
    print(f"Min Efficiency: {summary['Droplet_Separation_Efficiency']['min']*100:.1f}%")
    print(f"Max Efficiency: {summary['Droplet_Separation_Efficiency']['max']*100:.1f}%")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main() 