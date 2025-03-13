import numpy as np
import matplotlib.pyplot as plt
from ko_drum_physics_properties import KODrumPhysicalProperties
from ko_drum_parametric import KODrumParametricStudy

def main():
    # Base case parameters
    T_base = 293.15  # K (20°C)
    P_base = 101325  # Pa (1 atm)
    feed = {'H2O': 1.0}  # Pure water
    M_W = {'H2O': 18.015}  # Molecular weight of water
    
    # Create physics calculator
    physics = KODrumPhysicalProperties(T_base, P_base, feed, M_W)
    
    # Create parametric study with required arguments
    study = KODrumParametricStudy(physics, feed, M_W)
    
    # Temperature range for water composition analysis (20°C to 70°C)
    temperatures = np.linspace(293.15, 343.15, 50)  # 50 points from 20°C to 70°C
    
    print("Running water composition analysis...")
    results = physics.analyze_water_phase_diagram(temperatures)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Phase composition plot
    ax1.plot(results['temperature_C'], results['liquid_percent'], 'b-', label='Liquid')
    ax1.plot(results['temperature_C'], results['vapor_percent'], 'r-', label='Vapor')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Phase Percentage (%)')
    ax1.set_title('Water Phase Composition')
    ax1.grid(True)
    ax1.legend()
    
    # Vapor pressure plot
    ax2.plot(results['temperature_C'], results['vapor_pressure_bar'], 'r-')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Vapor Pressure (bar)')
    ax2.set_title('Water Vapor Pressure')
    ax2.grid(True)
    
    # Mass distribution plot
    ax3.plot(results['temperature_C'], results['mass_liquid_kg'], 'b-', label='Liquid')
    ax3.plot(results['temperature_C'], results['mass_vapor_kg'], 'r-', label='Vapor')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Mass (kg)')
    ax3.set_title('Mass Distribution')
    ax3.grid(True)
    ax3.legend()
    
    # Density plot
    ax4.plot(results['temperature_C'], results['liquid_density_kg_m3'], 'b-', label='Liquid')
    ax4.plot(results['temperature_C'], results['vapor_density_kg_m3'], 'r-', label='Vapor')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Density (kg/m³)')
    ax4.set_title('Phase Densities')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results table
    print("\nDetailed Results:")
    print(results.round(3))

if __name__ == "__main__":
    main() 