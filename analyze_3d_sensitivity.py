import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ko_drum_physics import KODrumPhysics
import pandas as pd
from ko_drum_config import OPERATING_CONDITIONS, VESSEL_DIMENSIONS

def analyze_3d_sensitivity():
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
    
    # Define ranges for height and diameter
    heights = np.linspace(base_height * 0.5, base_height * 2.0, 30)  # Increased resolution
    diameters = np.linspace(base_diameter * 0.5, base_diameter * 1.5, 30)  # Increased resolution
    
    # Create meshgrid for 3D surface
    H, D = np.meshgrid(heights, diameters)
    water_recovery = np.zeros_like(H)
    natural_recovery = np.zeros_like(H)
    mechanical_recovery = np.zeros_like(H)
    gas_velocities = np.zeros_like(H)
    residence_times = np.zeros_like(H)
    ld_ratios = H/D
    
    # Create mask for valid L/D ratios (between 2 and 5)
    valid_ld_mask = (ld_ratios >= 2) & (ld_ratios <= 5)
    
    print("\nAnalyzing sensitivity to height and diameter changes...")
    print(f"Height range: {min(heights):.1f}m to {max(heights):.1f}m")
    print(f"Diameter range: {min(diameters):.2f}m to {max(diameters):.2f}m")
    print("L/D ratio constraints: 2 ≤ L/D ≤ 5")
    
    # Calculate recovery for each combination
    for i, diameter in enumerate(diameters):
        for j, height in enumerate(heights):
            L_D_ratio = height / diameter
            
            # Skip calculations if L/D ratio is outside constraints
            if L_D_ratio < 2 or L_D_ratio > 5:
                water_recovery[i,j] = np.nan
                natural_recovery[i,j] = np.nan
                mechanical_recovery[i,j] = np.nan
                gas_velocities[i,j] = np.nan
                residence_times[i,j] = np.nan
                continue
                
            # Update dimensions
            ko_drum.diameter = diameter
            ko_drum.height = height
            
            # Calculate gas velocity and flow rates
            v_gas, v_max, Q_gas = ko_drum.calculate_gas_velocity()
            
            # Calculate settling velocities
            settling_velocities = ko_drum.calculate_settling_velocities()
            mean_settling_velocity = np.mean(settling_velocities)
            
            # Calculate residence time
            A_vessel = np.pi * (diameter/2)**2
            residence_time = (A_vessel * height) / Q_gas
            
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
            
            # Calculate water recovery
            mechanical_separation = droplet_efficiency * L_D_effect
            vapor_fraction = y_water
            total_recovery = natural_liquid_fraction + (vapor_fraction * mechanical_separation)
            
            # Store results
            water_recovery[i,j] = total_recovery * 100
            natural_recovery[i,j] = natural_liquid_fraction * 100
            mechanical_recovery[i,j] = (vapor_fraction * mechanical_separation) * 100
            gas_velocities[i,j] = v_gas
            residence_times[i,j] = residence_time
    
    # Find optimal point (ignoring NaN values)
    valid_recoveries = np.ma.array(water_recovery, mask=np.isnan(water_recovery))
    max_idx = np.unravel_index(np.ma.argmax(valid_recoveries), water_recovery.shape)
    opt_diameter = diameters[max_idx[0]]
    opt_height = heights[max_idx[1]]
    opt_recovery = water_recovery[max_idx]
    opt_ld = opt_height/opt_diameter
    
    print("\nOptimal Configuration (within L/D constraints):")
    print("-------------------------------------------")
    print(f"Optimal Diameter: {opt_diameter:.2f} m")
    print(f"Optimal Height: {opt_height:.2f} m")
    print(f"Optimal L/D Ratio: {opt_ld:.2f}")
    print(f"Maximum Water Recovery: {opt_recovery:.2f}%")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    
    # 3D Surface Plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(D, H, water_recovery, cmap='viridis')
    ax1.set_xlabel('Diameter (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_zlabel('Water Recovery (%)')
    plt.colorbar(surf, ax=ax1, label='Water Recovery (%)')
    
    # Add optimal point
    ax1.scatter([opt_diameter], [opt_height], [opt_recovery], 
                color='red', s=100, label='Optimal Point')
    
    # Add base case point if it's within L/D constraints
    base_ld = base_height/base_diameter
    if 2 <= base_ld <= 5:
        base_idx = (np.where(diameters >= base_diameter)[0][0],
                   np.where(heights >= base_height)[0][0])
        ax1.scatter([base_diameter], [base_height], 
                   [water_recovery[base_idx]], 
                   color='black', s=100, label='Base Case')
    
    ax1.legend()
    ax1.set_title('3D Water Recovery Surface\n(2 ≤ L/D ≤ 5)')
    
    # Contour Plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(D, H, water_recovery, levels=15, colors='black', linewidths=0.5)
    contourf = ax2.contourf(D, H, water_recovery, levels=15, cmap='viridis')
    plt.colorbar(contourf, ax=ax2, label='Water Recovery (%)')
    
    # Add optimal point to contour
    ax2.plot(opt_diameter, opt_height, 'r*', markersize=15, label='Optimal Point')
    
    # Add base case point to contour if within constraints
    if 2 <= base_ld <= 5:
        ax2.plot(base_diameter, base_height, 'k*', markersize=15, label='Base Case')
    
    # Add L/D ratio lines (only showing 2-5)
    ld_values = [2, 3, 4, 5]
    for ld in ld_values:
        d_range = np.linspace(min(diameters), max(diameters), 100)
        h_range = ld * d_range
        mask = (h_range >= min(heights)) & (h_range <= max(heights))
        if any(mask):
            ax2.plot(d_range[mask], h_range[mask], '--', color='gray', alpha=0.5)
            # Add L/D label at the middle of the line
            mid_idx = len(d_range[mask])//2
            ax2.annotate(f'L/D = {ld}', 
                        (d_range[mask][mid_idx], h_range[mask][mid_idx]),
                        xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Diameter (m)')
    ax2.set_ylabel('Height (m)')
    ax2.legend()
    ax2.set_title('Water Recovery Contour Map\nwith L/D Ratio Constraints (2 ≤ L/D ≤ 5)')
    ax2.grid(True)
    
    plt.suptitle('KO Drum Height-Diameter Sensitivity Analysis\n' + 
                f'(T = {T_C:.1f}°C, P = {P_bar:.3f} bar)',
                y=1.02)
    plt.tight_layout()
    
    # Create additional plots for mechanical and natural recovery
    fig2 = plt.figure(figsize=(15, 5))
    
    # Natural Recovery Contour
    ax3 = fig2.add_subplot(131)
    cf3 = ax3.contourf(D, H, natural_recovery, levels=15, cmap='viridis')
    plt.colorbar(cf3, ax=ax3, label='Natural Recovery (%)')
    ax3.set_xlabel('Diameter (m)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Natural Recovery\n(2 ≤ L/D ≤ 5)')
    
    # Mechanical Recovery Contour
    ax4 = fig2.add_subplot(132)
    cf4 = ax4.contourf(D, H, mechanical_recovery, levels=15, cmap='viridis')
    plt.colorbar(cf4, ax=ax4, label='Mechanical Recovery (%)')
    ax4.set_xlabel('Diameter (m)')
    ax4.set_ylabel('Height (m)')
    ax4.set_title('Mechanical Recovery\n(2 ≤ L/D ≤ 5)')
    
    # Gas Velocity Contour
    ax5 = fig2.add_subplot(133)
    cf5 = ax5.contourf(D, H, gas_velocities, levels=15, cmap='viridis')
    plt.colorbar(cf5, ax=ax5, label='Gas Velocity (m/s)')
    ax5.set_xlabel('Diameter (m)')
    ax5.set_ylabel('Height (m)')
    ax5.set_title('Gas Velocity\n(2 ≤ L/D ≤ 5)')
    
    plt.suptitle('Component Analysis (with L/D Constraints)', y=1.02)
    plt.tight_layout()
    
    return fig, fig2

if __name__ == "__main__":
    # Run 3D sensitivity analysis
    fig1, fig2 = analyze_3d_sensitivity()
    plt.show() 