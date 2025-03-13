import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ko_drum_physics_properties import KODrumPhysicalProperties
from ko_drum_simulation import KODrumSimulation

class KODrumParametricStudy:
    def __init__(self, operating_conditions, feed_composition, molecular_weights):
        self.operating_conditions = operating_conditions
        self.feed_composition = feed_composition
        self.molecular_weights = molecular_weights
        
        # Droplet properties
        self.min_droplet_size = 5e-6  # 5 microns
        self.max_droplet_size = 100e-6  # 100 microns
        self.n_droplet_sizes = 20
        self.droplet_sizes = np.linspace(self.min_droplet_size, self.max_droplet_size, self.n_droplet_sizes)
    
    def calculate_performance(self, height, diameter):
        """Calculate KO drum performance for given dimensions"""
        vessel_dimensions = {
            'height': height,
            'diameter': diameter,
            'liquid_holdup_height': 0.1 * height  # Assuming 10% liquid holdup
        }
        
        # Create simulation object
        simulation = KODrumSimulation(
            self.operating_conditions,
            self.feed_composition,
            self.molecular_weights,
            vessel_dimensions
        )
        
        # Run simulation
        results = simulation.simulate()
        
        # Extract key performance indicators
        performance = {
            'gas_velocity': results['gas_velocity'],
            'efficiency': results['separation_efficiency'],
            'L_D_ratio': results['L_D_ratio'],
            'water_recovery': results['water_recovery'],
            'pressure_drop': results['pressure_drop']['total_drop'],
            'energy_balance_error': results['energy_balance']['balance_error'],
            'mass_balance_error': results['mass_balance']['error_percent']
        }
        
        # Add validation flags
        performance['is_valid'] = (
            results['mass_balance']['is_valid'] and
            performance['pressure_drop'] < 5000 and  # 5000 Pa max pressure drop
            performance['gas_velocity'] < results['max_velocity']
        )
        
        return performance
    
    def height_parametric_study(self, heights, diameter):
        """Perform parametric study varying height"""
        results = []
        for height in heights:
            performance = self.calculate_performance(height, diameter)
            results.append({
                'Height (m)': height,
                'L/D': performance['L_D_ratio'],
                'Water Recovery (%)': performance['water_recovery'],
                'Gas Velocity (m/s)': performance['gas_velocity'],
                'Separation Efficiency (%)': performance['efficiency']
            })
        return pd.DataFrame(results)
    
    def multi_diameter_study(self, heights, diameters):
        """Perform parametric study for multiple diameters"""
        all_results = {}
        for diameter in diameters:
            results = self.height_parametric_study(heights, diameter)
            all_results[f'D={diameter:.2f}m'] = results
        return all_results
    
    def plot_multi_diameter_results(self, heights, diameters):
        """Create comparative plots for multiple diameters with enhanced physics"""
        results = self.multi_diameter_study(heights, diameters)
        
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Water Recovery
        ax2 = fig.add_subplot(gs[0, 1])  # Gas Velocity
        ax3 = fig.add_subplot(gs[1, 0])  # L/D Ratio
        ax4 = fig.add_subplot(gs[1, 1])  # Pressure Drop
        ax5 = fig.add_subplot(gs[2, 0])  # Energy Balance Error
        ax6 = fig.add_subplot(gs[2, 1])  # Mass Balance Error
        
        # Color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(diameters)))
        
        for (label, df), color in zip(results.items(), colors):
            # Plot results
            ax1.plot(df['Height (m)'], df['Water Recovery (%)'],
                    label=label, color=color)
            ax2.plot(df['Height (m)'], df['Gas Velocity (m/s)'],
                    label=label, color=color)
            ax3.plot(df['Height (m)'], df['L/D'],
                    label=label, color=color)
            ax4.plot(df['Height (m)'], df['Pressure Drop (Pa)'],
                    label=label, color=color)
            ax5.plot(df['Height (m)'], df['Energy Balance Error (J/hr)'],
                    label=label, color=color)
            ax6.plot(df['Height (m)'], df['Mass Balance Error (%)'],
                    label=label, color=color)
            
            # Add validity markers
            valid_points = df[df['is_valid']]
            ax1.scatter(valid_points['Height (m)'], valid_points['Water Recovery (%)'],
                       color=color, marker='o', s=50, alpha=0.5)
        
        # Customize plots
        ax1.set_xlabel('Vessel Height (m)')
        ax1.set_ylabel('Water Recovery (%)')
        ax1.set_title('Water Recovery vs Height')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_xlabel('Vessel Height (m)')
        ax2.set_ylabel('Gas Velocity (m/s)')
        ax2.set_title('Gas Velocity vs Height')
        ax2.grid(True)
        ax2.legend()
        
        ax3.set_xlabel('Vessel Height (m)')
        ax3.set_ylabel('L/D Ratio')
        ax3.set_title('L/D Ratio vs Height')
        ax3.grid(True)
        ax3.legend()
        
        ax4.set_xlabel('Vessel Height (m)')
        ax4.set_ylabel('Pressure Drop (Pa)')
        ax4.set_title('Pressure Drop vs Height')
        ax4.grid(True)
        ax4.legend()
        
        ax5.set_xlabel('Vessel Height (m)')
        ax5.set_ylabel('Energy Balance Error (J/hr)')
        ax5.set_title('Energy Balance Error vs Height')
        ax5.grid(True)
        ax5.legend()
        
        ax6.set_xlabel('Vessel Height (m)')
        ax6.set_ylabel('Mass Balance Error (%)')
        ax6.set_title('Mass Balance Error vs Height')
        ax6.grid(True)
        ax6.legend()
        
        plt.suptitle(f'Enhanced KO Drum Performance Analysis\nT = {self.operating_conditions["temperature_K"]-273.15:.1f}°C, P = {self.operating_conditions["pressure_Pa"]/1e5:.1f} bar')
        plt.tight_layout()
        return fig
    
    def temperature_parametric_study(self, temperatures, height, diameter):
        """Perform parametric study varying temperature"""
        original_T = self.operating_conditions["temperature_K"]
        results = []
        
        for T in temperatures:
            # Update temperature
            self.operating_conditions["temperature_K"] = T
            
            # Calculate performance
            performance = self.calculate_performance(height, diameter)
            
            results.append({
                'Temperature (°C)': T - 273.15,
                'Water Recovery (%)': performance['water_recovery'],
                'Gas Velocity (m/s)': performance['gas_velocity'],
                'Separation Efficiency (%)': performance['efficiency']
            })
        
        # Restore original temperature
        self.operating_conditions["temperature_K"] = original_T
        
        return pd.DataFrame(results)
    
    def plot_temperature_study(self, temperatures, height, diameter):
        """Create plots for temperature parametric study"""
        results = self.temperature_parametric_study(temperatures, height, diameter)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot water recovery and separation efficiency
        ax1.plot(results['Temperature (°C)'], results['Water Recovery (%)'], 'b-', label='Water Recovery')
        ax1.plot(results['Temperature (°C)'], results['Separation Efficiency (%)'], 'r--', label='Separation Efficiency')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Efficiency (%)')
        ax1.set_title('Recovery and Efficiency vs Temperature')
        ax1.grid(True)
        ax1.legend()
        
        # Plot gas velocity
        ax2.plot(results['Temperature (°C)'], results['Gas Velocity (m/s)'], 'g-')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Gas Velocity (m/s)')
        ax2.set_title('Gas Velocity vs Temperature')
        ax2.grid(True)
        
        plt.suptitle(f'Temperature Effects on KO Drum Performance\nH = {height:.1f}m, D = {diameter:.2f}m, P = {self.operating_conditions["pressure_Pa"]/1e5:.1f} bar')
        plt.tight_layout()
        return fig
    
    def comprehensive_study(self, heights, diameters, temperatures):
        """Perform comprehensive parametric study across heights, diameters, and temperatures"""
        results = []
        
        for T in temperatures:
            # Update temperature
            self.operating_conditions["temperature_K"] = T
            
            for D in diameters:
                for H in heights:
                    performance = self.calculate_performance(H, D)
                    results.append({
                        'Temperature (°C)': T - 273.15,
                        'Height (m)': H,
                        'Diameter (m)': D,
                        'L/D': H/D,
                        'Water Recovery (%)': performance['water_recovery'],
                        'Gas Velocity (m/s)': performance['gas_velocity'],
                        'Separation Efficiency (%)': performance['efficiency'],
                        'Pressure Drop (Pa)': performance['pressure_drop'],
                        'Energy Balance Error (J/hr)': performance['energy_balance_error'],
                        'Mass Balance Error (%)': performance['mass_balance_error'],
                        'is_valid': performance['is_valid']
                    })
        
        # Restore original temperature
        self.operating_conditions["temperature_K"] = self.operating_conditions["temperature_K"]
        
        return pd.DataFrame(results)
    
    def plot_comprehensive_results(self, heights, diameters, temperatures):
        """Create enhanced 3D surface plots for comprehensive study results"""
        results = self.comprehensive_study(heights, diameters, temperatures)
        
        fig = plt.figure(figsize=(15, 15))
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # Create subplots
        plots = [
            ('Water Recovery (%)', 'Water Recovery (%)', (0, 0)),
            ('Gas Velocity (m/s)', 'Gas Velocity (m/s)', (0, 1)),
            ('L/D Ratio', 'L/D', (1, 0)),
            ('Pressure Drop (Pa)', 'Pressure Drop (Pa)', (1, 1)),
            ('Energy Balance Error (J/hr)', 'Energy Balance Error (J/hr)', (2, 0)),
            ('Mass Balance Error (%)', 'Mass Balance Error (%)', (2, 1))
        ]
        
        for title, key, pos in plots:
            ax = fig.add_subplot(gs[pos], projection='3d')
            scatter = ax.scatter(results['Height (m)'],
                               results['Diameter (m)'],
                               results[key],
                               c=results['Temperature (°C)'],
                               cmap='viridis')
            ax.set_xlabel('Height (m)')
            ax.set_ylabel('Diameter (m)')
            ax.set_zlabel(title)
            plt.colorbar(scatter, ax=ax, label='Temperature (°C)')
            ax.set_title(title)
            
            # Add validity markers if available
            if 'is_valid' in results.columns:
                valid_points = results[results['is_valid']]
                ax.scatter(valid_points['Height (m)'],
                          valid_points['Diameter (m)'],
                          valid_points[key],
                          c='g', marker='*', s=100, alpha=0.5,
                          label='Valid Design Points')
                ax.legend()
        
        plt.suptitle('Enhanced KO Drum Performance Analysis\nIncluding Energy and Mass Balance')
        plt.tight_layout()
        return fig

    def analyze_water_composition(self, temperatures):
        """Analyze water vapor-liquid composition across temperature range"""
        results = []
        
        for T in temperatures:
            # Update temperature
            self.operating_conditions["temperature_K"] = T
            
            # Create physics calculator
            physics = KODrumPhysicalProperties(
                self.operating_conditions["temperature_K"],
                self.operating_conditions["pressure_Pa"],
                self.feed_composition,
                self.molecular_weights
            )
            
            # Calculate phase properties
            phase_props = physics.calculate_phase_properties()
            
            results.append({
                'Temperature (°C)': phase_props['temperature_C'],
                'Vapor Fraction': phase_props['vapor_phase']['composition'],
                'Liquid Fraction': phase_props['liquid_phase']['composition'],
                'Vapor Pressure (bar)': phase_props['vapor_phase']['pressure'] / 1e5,
                'Vapor Density (kg/m³)': phase_props['vapor_phase']['density'],
                'Liquid Density (kg/m³)': phase_props['liquid_phase']['density']
            })
        
        return pd.DataFrame(results)

    def plot_water_composition(self, temperatures):
        """Plot water composition analysis results"""
        results = self.analyze_water_composition(temperatures)
        
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Composition plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['Temperature (°C)'], results['Vapor Fraction'], 'r-', label='Vapor')
        ax1.plot(results['Temperature (°C)'], results['Liquid Fraction'], 'b-', label='Liquid')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Phase Fraction')
        ax1.set_title('Water Phase Composition')
        ax1.grid(True)
        ax1.legend()
        
        # Vapor pressure plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results['Temperature (°C)'], results['Vapor Pressure (bar)'], 'g-')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Vapor Pressure (bar)')
        ax2.set_title('Water Vapor Pressure')
        ax2.grid(True)
        
        # Density plot
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(results['Temperature (°C)'], results['Vapor Density (kg/m³)'], 'r--', label='Vapor')
        ax3.plot(results['Temperature (°C)'], results['Liquid Density (kg/m³)'], 'b--', label='Liquid')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Density (kg/m³)')
        ax3.set_title('Phase Densities')
        ax3.grid(True)
        ax3.legend()
        
        plt.suptitle(f'Water Phase Behavior Analysis\nP = {self.operating_conditions["pressure_Pa"]/1e5:.1f} bar')
        plt.tight_layout()
        
        # Add results table
        table_data = results.round(3)
        print("\nWater Composition Analysis Results:")
        print(table_data.to_string(index=False))
        
        return fig

    def plot_water_phase_diagram(self, temperatures, total_water_mass=1.0, system_volume=1.0):
        """Create detailed water phase composition diagram"""
        # Create physics calculator
        physics = KODrumPhysicalProperties(
            self.operating_conditions["temperature_K"],
            self.operating_conditions["pressure_Pa"],
            self.feed_composition,
            self.molecular_weights
        )
        
        # Get phase composition data
        results = physics.analyze_water_phase_diagram(temperatures, total_water_mass, system_volume)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Phase composition plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['temperature_C'], results['vapor_percent'], 'r-', label='Vapor Phase')
        ax1.plot(results['temperature_C'], results['liquid_percent'], 'b-', label='Liquid Phase')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Phase Composition (%)')
        ax1.set_title('Water Phase Composition vs Temperature')
        ax1.grid(True)
        ax1.legend()
        
        # Vapor pressure plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results['temperature_C'], results['vapor_pressure_bar'], 'g-')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Vapor Pressure (bar)')
        ax2.set_title('Water Vapor Pressure vs Temperature')
        ax2.grid(True)
        
        # Mass distribution plot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(results['temperature_C'], results['mass_vapor_kg'], 'r--', label='Vapor Mass')
        ax3.plot(results['temperature_C'], results['mass_liquid_kg'], 'b--', label='Liquid Mass')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Mass (kg)')
        ax3.set_title('Phase Mass Distribution')
        ax3.grid(True)
        ax3.legend()
        
        # Density plot
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(results['temperature_C'], results['vapor_density_kg_m3'], 'r:', label='Vapor Density')
        ax4.plot(results['temperature_C'], results['liquid_density_kg_m3'], 'b:', label='Liquid Density')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Density (kg/m³)')
        ax4.set_title('Phase Densities vs Temperature')
        ax4.grid(True)
        ax4.legend()
        
        plt.suptitle(f'Detailed Water Phase Diagram Analysis\nTotal Mass: {total_water_mass:.2f} kg, System Volume: {system_volume:.2f} m³')
        plt.tight_layout()
        
        # Print detailed results table
        print("\nDetailed Water Phase Composition Analysis:")
        print(results.round(3).to_string(index=False))
        
        return fig

if __name__ == "__main__":
    from ko_drum_config import (
        OPERATING_CONDITIONS,
        FEED_COMPOSITION,
        MOLECULAR_WEIGHTS
    )
    
    # Create parametric study object
    study = KODrumParametricStudy(
        OPERATING_CONDITIONS,
        FEED_COMPOSITION,
        MOLECULAR_WEIGHTS
    )
    
    # Define study ranges
    heights = np.linspace(4.0, 16.0, 10)
    diameters = np.linspace(1.3, 1.7, 5)
    temperatures = np.linspace(293.15, 353.15, 5)  # 20°C to 80°C
    
    # Run comprehensive parametric study
    print("\nRunning Comprehensive Parametric Study...")
    fig = study.plot_comprehensive_results(heights, diameters, temperatures)
    plt.show()
    
    # Run temperature study for base case
    print("\nRunning Temperature Sensitivity Study...")
    base_height = 5.0
    base_diameter = 1.45
    temp_fig = study.plot_temperature_study(temperatures, base_height, base_diameter)
    plt.show()
    
    # Analyze water composition from 20°C to 70°C
    temperatures = np.linspace(293.15, 343.15, 11)  # 20°C to 70°C in 5°C steps
    
    print("\nAnalyzing Water Composition...")
    fig = study.plot_water_composition(temperatures)
    plt.show()
    
    # Create detailed water phase diagram
    print("\nGenerating Detailed Water Phase Diagram...")
    phase_diagram = study.plot_water_phase_diagram(
        temperatures,
        total_water_mass=1.0,  # 1 kg of water
        system_volume=1.0      # 1 m³ system volume
    )
    plt.show() 