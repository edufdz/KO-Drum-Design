import numpy as np
import matplotlib.pyplot as plt
from ko_drum_physics import KODrumPhysics
import pandas as pd
from ko_drum_config import OPERATING_CONDITIONS

def analyze_height_sensitivity():
    # Create KO Drum simulator
    ko_drum = KODrumPhysics()
    
    # Calculate initial vapor-liquid equilibrium
    T = OPERATING_CONDITIONS['temperature_K']
    T_C = OPERATING_CONDITIONS['temperature']
    P_bar = OPERATING_CONDITIONS['pressure']
    P_Pa = OPERATING_CONDITIONS['pressure_Pa']
    
    A, B, C = 8.07131, 1730.63, 233.426
    P_vap = 133.322 * 10**(A - B/(T_C + C))  # Pa
    y_water = min(P_vap / P_Pa, 1.0)
    natural_liquid_fraction = 1 - y_water
    
    print("\nInitial Conditions and Physical Properties:")
    print("----------------------------------------")
    print(f"Temperature: {T_C:.1f}°C")
    print(f"Pressure: {P_bar:.2f} bar")
    print(f"Base Diameter: {ko_drum.diameter:.2f} m")
    print(f"Base Height: {ko_drum.height:.2f} m")
    print(f"Initial L/D Ratio: {ko_drum.height/ko_drum.diameter:.2f}")
    print(f"\nVapor Pressure: {P_vap/1e5:.3f} bar")
    print(f"Natural Vapor Fraction: {y_water:.3f}")
    print(f"Natural Liquid Fraction: {natural_liquid_fraction:.3f}")
    
    # Calculate fluid properties
    ko_drum.calculate_fluid_properties()
    rho_gas = ko_drum.fluid_properties['gas']['density']
    rho_liquid = ko_drum.fluid_properties['liquid']['density']
    mu_gas = ko_drum.fluid_properties['gas']['viscosity']
    
    print("\nPhysical Properties:")
    print(f"Gas Density: {rho_gas:.2f} kg/m³")
    print(f"Liquid Density: {rho_liquid:.2f} kg/m³")
    print(f"Gas Viscosity: {mu_gas*1e6:.2f} μPa·s")
    
    # Define height range for analysis (0.5x to 2x of original height)
    original_height = ko_drum.height
    heights = np.linspace(original_height * 0.5, original_height * 2.0, 20)
    
    print("\nAnalyzing water recovery sensitivity to height...")
    print(f"Height range: {min(heights):.1f}m to {max(heights):.1f}m")
    
    # Initialize results storage
    results_data = []
    
    # Calculate base gas velocity and flow rates
    v_gas_base, v_max, Q_gas = ko_drum.calculate_gas_velocity()
    
    for height in heights:
        ko_drum.height = height
        L_D_ratio = height / ko_drum.diameter
        
        # Calculate settling velocities for droplet distribution
        settling_velocities = ko_drum.calculate_settling_velocities()
        
        # Calculate residence time
        residence_time = height / v_gas_base
        
        # Calculate droplet separation efficiency for different sizes
        droplet_efficiencies = []
        for d in ko_drum.droplet_sizes:
            eff = ko_drum.calculate_mist_eliminator_efficiency(d, v_gas_base)
            droplet_efficiencies.append(eff)
        
        # Weight efficiencies by log-normal droplet size distribution
        log_mean = np.log(np.mean(ko_drum.droplet_sizes))
        log_std = 0.35
        weights = np.exp(-(np.log(ko_drum.droplet_sizes) - log_mean)**2 / (2 * log_std**2))
        weights = weights / np.sum(weights)
        
        # Calculate overall droplet separation efficiency
        droplet_efficiency = np.sum(np.array(droplet_efficiencies) * weights)
        
        # Calculate mean settling velocity
        mean_settling_velocity = np.mean(settling_velocities)
        
        # Calculate separation parameter
        A_vessel = np.pi * (ko_drum.diameter/2)**2
        separation_parameter = (height * A_vessel / Q_gas) * (mean_settling_velocity / height)
        
        # Calculate L/D effect on efficiency
        L_D_effect = np.tanh(0.5 * L_D_ratio)
        
        # Calculate combined water recovery
        mechanical_separation = droplet_efficiency * L_D_effect
        vapor_fraction = y_water
        total_recovery = natural_liquid_fraction + (vapor_fraction * mechanical_separation)
        
        # Store results
        results_data.append({
            'Height (m)': height,
            'L/D': L_D_ratio,
            'Water Recovery (%)': total_recovery * 100,
            'Natural Recovery (%)': natural_liquid_fraction * 100,
            'Mechanical Recovery (%)': (vapor_fraction * mechanical_separation) * 100,
            'Gas Velocity (m/s)': v_gas_base,
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
    print("\nHeight (m) | L/D Ratio | Natural (%) | Mechanical (%) | Total Recovery (%) | Gas Velocity (m/s)")
    print("-" * 95)
    for _, row in results.iterrows():
        print(f"{row['Height (m)']:9.2f} | {row['L/D']:9.2f} | {row['Natural Recovery (%)']:11.2f} | "
              f"{row['Mechanical Recovery (%)']:13.2f} | {row['Water Recovery (%)']:16.2f} | {row['Gas Velocity (m/s)']:16.2f}")
    
    # Calculate statistics
    print("\nStatistical Summary:")
    print("------------------")
    print(f"Optimal Height: {results.loc[results['Water Recovery (%)'].idxmax(), 'Height (m)']:.2f} m")
    print(f"Optimal L/D Ratio: {results.loc[results['Water Recovery (%)'].idxmax(), 'L/D']:.2f}")
    print(f"Maximum Total Recovery: {results['Water Recovery (%)'].max():.2f}%")
    print(f"- Natural Recovery: {results.loc[results['Water Recovery (%)'].idxmax(), 'Natural Recovery (%)']:.2f}%")
    print(f"- Mechanical Recovery: {results.loc[results['Water Recovery (%)'].idxmax(), 'Mechanical Recovery (%)']:.2f}%")
    
    # Create enhanced visualization
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    # Water Recovery Components vs Height
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['Height (m)'], results['Natural Recovery (%)'], 'b-', 
            label='Natural Recovery', linewidth=2, alpha=0.7)
    ax1.plot(results['Height (m)'], results['Mechanical Recovery (%)'], 'g-', 
            label='Mechanical Recovery', linewidth=2, alpha=0.7)
    ax1.plot(results['Height (m)'], results['Water Recovery (%)'], 'r-', 
            label='Total Recovery', linewidth=2)
    ax1.set_xlabel('Vessel Height (m)')
    ax1.set_ylabel('Recovery (%)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Water Recovery Components vs Height')
    
    # L/D Ratio Effects
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['L/D'], results['Water Recovery (%)'], 'b-', linewidth=2)
    ax2.set_xlabel('L/D Ratio')
    ax2.set_ylabel('Total Water Recovery (%)')
    ax2.grid(True)
    ax2.set_title('Water Recovery vs L/D Ratio')
    
    # Separation Parameters
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['Height (m)'], results['Droplet Efficiency (%)'], 'b-', 
            label='Droplet Efficiency', linewidth=2)
    ax3.plot(results['Height (m)'], results['L/D Effect'] * 100, 'r--', 
            label='L/D Effect', linewidth=2)
    ax3.set_xlabel('Vessel Height (m)')
    ax3.set_ylabel('Efficiency (%)')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Separation Parameters vs Height')
    
    # Gas Velocity and Residence Time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    
    ax4.plot(results['Height (m)'], results['Gas Velocity (m/s)'], 'b-', 
            label='Gas Velocity', linewidth=2)
    ax4.axhline(y=v_max, color='r', linestyle='--', label='Maximum Velocity')
    ax4.set_xlabel('Vessel Height (m)')
    ax4.set_ylabel('Gas Velocity (m/s)', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    ax4_twin.plot(results['Height (m)'], results['Residence Time (s)'], 'g-', 
                label='Residence Time', linewidth=2)
    ax4_twin.set_ylabel('Residence Time (s)', color='g')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2)
    ax4.set_title('Operating Parameters vs Height')
    
    plt.suptitle('KO Drum Height Sensitivity Analysis\n' + 
                f'(T = {T_C:.1f}°C, P = {P_bar:.3f} bar, D = {ko_drum.diameter:.2f}m)',
                y=1.02)
    plt.tight_layout()
    
    return fig, results

if __name__ == "__main__":
    # Run height sensitivity analysis
    fig, results = analyze_height_sensitivity()
    plt.show() 