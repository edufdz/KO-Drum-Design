import numpy as np
import matplotlib.pyplot as plt
from ko_drum_physics import KODrumPhysics
import pandas as pd
from ko_drum_config import OPERATING_CONDITIONS, VESSEL_DIMENSIONS

def analyze_diameter_sensitivity():
    # Create KO Drum simulator
    ko_drum = KODrumPhysics()
    
    # Get base configuration values
    T = OPERATING_CONDITIONS['temperature_K']
    T_C = OPERATING_CONDITIONS['temperature']
    P_bar = OPERATING_CONDITIONS['pressure']
    P_Pa = OPERATING_CONDITIONS['pressure_Pa']
    base_diameter = VESSEL_DIMENSIONS['diameter']
    base_height = VESSEL_DIMENSIONS['height']
    
    # Calculate initial vapor-liquid equilibrium
    A, B, C = 8.07131, 1730.63, 233.426
    P_vap = 133.322 * 10**(A - B/(T_C + C))  # Pa
    y_water = min(P_vap / P_Pa, 1.0)
    natural_liquid_fraction = 1 - y_water
    
    # Calculate fluid properties first
    ko_drum.calculate_fluid_properties()
    rho_gas = ko_drum.fluid_properties['gas']['density']
    rho_liquid = ko_drum.fluid_properties['liquid']['density']
    mu_gas = ko_drum.fluid_properties['gas']['viscosity']
    
    print("\nInitial Conditions:")
    print("------------------")
    print(f"Temperature: {T_C:.1f}°C")
    print(f"Pressure: {P_bar:.3f} bar")
    print(f"Base Diameter: {base_diameter:.2f} m")
    print(f"Base Height: {base_height:.2f} m")
    print(f"Base L/D Ratio: {base_height/base_diameter:.2f}")
    print(f"Vapor Pressure: {P_vap/1e5:.3f} bar")
    print(f"\nFluid Properties:")
    print(f"Gas Density: {rho_gas:.2f} kg/m³")
    print(f"Liquid Density: {rho_liquid:.2f} kg/m³")
    print(f"Gas Viscosity: {mu_gas*1e6:.2f} μPa·s")
    
    # Define diameter range (0.5x to 1.5x of original diameter)
    diameters = np.linspace(base_diameter * 0.5, base_diameter * 1.5, 20)
    
    print("\nAnalyzing sensitivity to diameter changes...")
    print(f"Diameter range: {min(diameters):.2f}m to {max(diameters):.2f}m")
    
    # Initialize results storage
    results_data = []
    
    for diameter in diameters:
        # Update diameter
        ko_drum.diameter = diameter
        L_D_ratio = base_height / diameter
        
        # Calculate gas velocity and flow rates
        v_gas, v_max, Q_gas = ko_drum.calculate_gas_velocity()
        
        # Calculate settling velocities
        settling_velocities = ko_drum.calculate_settling_velocities()
        mean_settling_velocity = np.mean(settling_velocities)
        
        # Calculate residence time
        A_vessel = np.pi * (diameter/2)**2
        residence_time = (A_vessel * base_height) / Q_gas
        
        # Calculate droplet separation efficiency
        droplet_efficiencies = []
        for d in ko_drum.droplet_sizes:
            eff = ko_drum.calculate_mist_eliminator_efficiency(d, v_gas)
            droplet_efficiencies.append(eff)
        
        # Weight efficiencies by log-normal distribution
        log_mean = np.log(np.mean(ko_drum.droplet_sizes))
        log_std = 0.35
        weights = np.exp(-(np.log(ko_drum.droplet_sizes) - log_mean)**2 / (2 * log_std**2))
        weights = weights / np.sum(weights)
        droplet_efficiency = np.sum(np.array(droplet_efficiencies) * weights)
        
        # Calculate L/D effect
        L_D_effect = np.tanh(0.5 * L_D_ratio)
        
        # Calculate separation parameter
        separation_parameter = (base_height * A_vessel / Q_gas) * (mean_settling_velocity / base_height)
        
        # Calculate water recovery
        mechanical_separation = droplet_efficiency * L_D_effect
        vapor_fraction = y_water
        total_recovery = natural_liquid_fraction + (vapor_fraction * mechanical_separation)
        
        # Store results
        results_data.append({
            'Diameter (m)': diameter,
            'L/D': L_D_ratio,
            'Water Recovery (%)': total_recovery * 100,
            'Natural Recovery (%)': natural_liquid_fraction * 100,
            'Mechanical Recovery (%)': (vapor_fraction * mechanical_separation) * 100,
            'Gas Velocity (m/s)': v_gas,
            'Max Gas Velocity (m/s)': v_max,
            'Residence Time (s)': residence_time,
            'Droplet Efficiency (%)': droplet_efficiency * 100,
            'L/D Effect': L_D_effect,
            'Separation Parameter': separation_parameter
        })
    
    # Convert to DataFrame
    results = pd.DataFrame(results_data)
    
    # Print detailed results
    print("\nDetailed Analysis Results:")
    print("------------------------")
    print("\nDiameter (m) | L/D Ratio | Natural (%) | Mechanical (%) | Total Recovery (%) | Gas Velocity (m/s)")
    print("-" * 95)
    for _, row in results.iterrows():
        print(f"{row['Diameter (m)']:11.2f} | {row['L/D']:9.2f} | {row['Natural Recovery (%)']:11.2f} | "
              f"{row['Mechanical Recovery (%)']:13.2f} | {row['Water Recovery (%)']:16.2f} | {row['Gas Velocity (m/s)']:16.2f}")
    
    # Calculate statistics
    print("\nStatistical Summary:")
    print("------------------")
    print(f"Optimal Diameter: {results.loc[results['Water Recovery (%)'].idxmax(), 'Diameter (m)']:.2f} m")
    print(f"Optimal L/D Ratio: {results.loc[results['Water Recovery (%)'].idxmax(), 'L/D']:.2f}")
    print(f"Maximum Total Recovery: {results['Water Recovery (%)'].max():.2f}%")
    print(f"- Natural Recovery: {results.loc[results['Water Recovery (%)'].idxmax(), 'Natural Recovery (%)']:.2f}%")
    print(f"- Mechanical Recovery: {results.loc[results['Water Recovery (%)'].idxmax(), 'Mechanical Recovery (%)']:.2f}%")
    
    # Create visualization
    fig = plt.figure(figsize=(15, 12))  # Made figure taller
    gs = plt.GridSpec(3, 2)  # Changed to 3x2 grid
    
    # Water Recovery Components vs Diameter (existing)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['Diameter (m)'], results['Natural Recovery (%)'], 'b-', 
            label='Natural Recovery', linewidth=2, alpha=0.7)
    ax1.plot(results['Diameter (m)'], results['Mechanical Recovery (%)'], 'g-', 
            label='Mechanical Recovery', linewidth=2, alpha=0.7)
    ax1.plot(results['Diameter (m)'], results['Water Recovery (%)'], 'r-', 
            label='Total Recovery', linewidth=2)
    ax1.axvline(x=base_diameter, color='k', linestyle='--', label='Base Diameter')
    ax1.set_xlabel('Vessel Diameter (m)')
    ax1.set_ylabel('Recovery (%)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Water Recovery Components vs Diameter')
    
    # L/D Ratio Effects (existing)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['L/D'], results['Water Recovery (%)'], 'b-', linewidth=2)
    ax2.axvline(x=base_height/base_diameter, color='k', linestyle='--', label='Base L/D')
    ax2.set_xlabel('L/D Ratio')
    ax2.set_ylabel('Total Water Recovery (%)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Water Recovery vs L/D Ratio')
    
    # Gas Velocity and Residence Time Correlation
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_twin = ax3.twinx()
    
    # Plot gas velocity
    ln1 = ax3.plot(results['Diameter (m)'], results['Gas Velocity (m/s)'], 'b-', 
                   label='Gas Velocity', linewidth=2)
    ax3.set_xlabel('Vessel Diameter (m)')
    ax3.set_ylabel('Gas Velocity (m/s)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    
    # Plot residence time
    ln2 = ax3_twin.plot(results['Diameter (m)'], results['Residence Time (s)'], 'r-', 
                        label='Residence Time', linewidth=2)
    ax3_twin.set_ylabel('Residence Time (s)', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    # Add combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs)
    ax3.set_title('Gas Velocity and Residence Time vs Diameter')
    ax3.grid(True)
    
    # Droplet Efficiency vs Diameter
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results['Diameter (m)'], results['Droplet Efficiency (%)'], 'b-', linewidth=2)
    ax4.set_xlabel('Vessel Diameter (m)')
    ax4.set_ylabel('Droplet Efficiency (%)')
    ax4.grid(True)
    ax4.set_title('Droplet Separation Efficiency vs Diameter')
    
    # Efficiency Correlation Plot
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(results['Gas Velocity (m/s)'], 
                         results['Droplet Efficiency (%)'],
                         c=results['Residence Time (s)'],
                         cmap='viridis',
                         s=100)
    ax5.set_xlabel('Gas Velocity (m/s)')
    ax5.set_ylabel('Droplet Efficiency (%)')
    ax5.grid(True)
    plt.colorbar(scatter, ax=ax5, label='Residence Time (s)')
    ax5.set_title('Efficiency vs Gas Velocity\nColored by Residence Time')
    
    # Performance Map
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(results['L/D'],
                         results['Droplet Efficiency (%)'],
                         c=results['Water Recovery (%)'],
                         cmap='viridis',
                         s=100)
    ax6.set_xlabel('L/D Ratio')
    ax6.set_ylabel('Droplet Efficiency (%)')
    ax6.grid(True)
    plt.colorbar(scatter, ax=ax6, label='Total Water Recovery (%)')
    ax6.set_title('Efficiency vs L/D Ratio\nColored by Water Recovery')
    
    plt.suptitle('KO Drum Diameter Sensitivity Analysis\n' + 
                f'(T = {T_C:.1f}°C, P = {P_bar:.3f} bar, H = {base_height:.2f}m)',
                y=1.02)
    plt.tight_layout()
    
    # Print correlation analysis
    print("\nCorrelation Analysis:")
    print("--------------------")
    correlations = results[['Diameter (m)', 'Gas Velocity (m/s)', 'Residence Time (s)', 
                          'Droplet Efficiency (%)', 'Water Recovery (%)', 'L/D']].corr()
    print("\nCorrelations with Droplet Efficiency:")
    print(correlations['Droplet Efficiency (%)'].sort_values(ascending=False))
    
    return fig, results

if __name__ == "__main__":
    # Run diameter sensitivity analysis
    fig, results = analyze_diameter_sensitivity()
    plt.show() 