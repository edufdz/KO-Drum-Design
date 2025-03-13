import numpy as np
from ko_drum_physics import KODrumPhysics
import pandas as pd
import matplotlib.pyplot as plt
from ko_drum_config import OPERATING_CONDITIONS, VESSEL_DIMENSIONS
from optimised_drum_config import (
    OPERATING_CONDITIONS as OPT_OPERATING_CONDITIONS,
    VESSEL_DIMENSIONS as OPT_VESSEL_DIMENSIONS
)
from CoolProp.CoolProp import PropsSI

def analyze_stream_properties(ko_drum, config_name="Base", operating_conditions=None):
    """Analyze mass and energy balances for a given KO drum configuration"""
    # Set operating conditions if provided
    if operating_conditions:
        ko_drum.T = operating_conditions['temperature_K']
        ko_drum.P = operating_conditions['pressure_Pa']
    
    # Run simulation
    results = ko_drum.simulate()
    
    # Get stream compositions
    feed = results['stream_results']['feed']
    gas_out = results['stream_results']['gas_outlet']
    liquid_out = results['stream_results']['liquid_outlet']
    
    # Calculate mass flows
    mass_flows = {
        'Feed': {},
        'Gas Outlet': {},
        'Liquid Outlet': {}
    }
    
    total_mass = {'Feed': 0, 'Gas Outlet': 0, 'Liquid Outlet': 0}
    
    # Calculate component mass flows
    for comp in feed:
        # Feed stream
        mass_flows['Feed'][comp] = feed[comp] * ko_drum.M_W[comp]
        total_mass['Feed'] += mass_flows['Feed'][comp]
        
        # Gas outlet
        mass_flows['Gas Outlet'][comp] = gas_out[comp] * ko_drum.M_W[comp]
        total_mass['Gas Outlet'] += mass_flows['Gas Outlet'][comp]
        
        # Liquid outlet
        mass_flows['Liquid Outlet'][comp] = liquid_out[comp] * ko_drum.M_W[comp]
        total_mass['Liquid Outlet'] += mass_flows['Liquid Outlet'][comp]
    
    # Calculate energy flows using the current operating conditions
    energy_flows = {
        'Feed': 0,
        'Gas Outlet': 0,
        'Liquid Outlet': 0
    }
    
    T = ko_drum.T  # Use actual temperature from simulator
    P = ko_drum.P  # Use actual pressure from simulator
    
    # Calculate enthalpies for each stream
    for comp in feed:
        try:
            h = PropsSI('H', 'T', T, 'P', P, comp)
            # Feed stream
            energy_flows['Feed'] += feed[comp] * h
            # Gas outlet
            energy_flows['Gas Outlet'] += gas_out[comp] * h
            # Liquid outlet
            energy_flows['Liquid Outlet'] += liquid_out[comp] * h
        except:
            print(f"Warning: Could not calculate enthalpy for {comp}")
    
    # Convert energy flows from J/s to kW
    energy_flows = {k: v/1000 for k, v in energy_flows.items()}
    
    # Calculate water recovery
    water_recovery = results['separation_performance']
    
    return {
        'Configuration': config_name,
        'Mass Flows': mass_flows,
        'Total Mass': total_mass,
        'Energy Flows': energy_flows,
        'Water Recovery': water_recovery,
        'Operating Parameters': results['operating_conditions'],
        'Operating Conditions': {
            'Temperature': T - 273.15,  # Convert to °C
            'Pressure': P / 1e5  # Convert to bar
        },
        'Vessel': {
            'Diameter': ko_drum.diameter,
            'Height': ko_drum.height
        }
    }

def print_detailed_results(results):
    """Print detailed analysis results"""
    print(f"\n{results['Configuration']} Configuration Analysis")
    print("=" * 50)
    
    print("\nOperating Conditions:")
    print("-" * 30)
    print(f"Temperature: {results['Operating Conditions']['Temperature']:.1f}°C")
    print(f"Pressure: {results['Operating Conditions']['Pressure']:.3f} bar")
    
    print("\nMass Balance (kg/h):")
    print("-" * 30)
    components = results['Mass Flows']['Feed'].keys()
    
    # Create DataFrame for mass flows
    mass_data = []
    for comp in components:
        mass_data.append({
            'Component': comp,
            'Feed': results['Mass Flows']['Feed'][comp],
            'Gas Out': results['Mass Flows']['Gas Outlet'][comp],
            'Liquid Out': results['Mass Flows']['Liquid Outlet'][comp]
        })
    
    mass_df = pd.DataFrame(mass_data)
    mass_df['Mass Balance'] = mass_df['Feed'] - (mass_df['Gas Out'] + mass_df['Liquid Out'])
    print(mass_df.round(2).to_string(index=False))
    
    print("\nTotal Mass Flows (kg/h):")
    print(f"Feed: {results['Total Mass']['Feed']:.2f}")
    print(f"Gas Outlet: {results['Total Mass']['Gas Outlet']:.2f}")
    print(f"Liquid Outlet: {results['Total Mass']['Liquid Outlet']:.2f}")
    print(f"Mass Balance: {results['Total Mass']['Feed'] - (results['Total Mass']['Gas Outlet'] + results['Total Mass']['Liquid Outlet']):.2f}")
    
    print("\nEnergy Balance (kW):")
    print("-" * 30)
    print(f"Feed Enthalpy: {results['Energy Flows']['Feed']:.2f}")
    print(f"Gas Outlet Enthalpy: {results['Energy Flows']['Gas Outlet']:.2f}")
    print(f"Liquid Outlet Enthalpy: {results['Energy Flows']['Liquid Outlet']:.2f}")
    print(f"Energy Balance: {results['Energy Flows']['Feed'] - (results['Energy Flows']['Gas Outlet'] + results['Energy Flows']['Liquid Outlet']):.2f}")
    
    print("\nWater Recovery Performance (%):")
    print("-" * 30)
    print(f"Natural Recovery: {results['Water Recovery']['natural_recovery']:.2f}%")
    print(f"Mechanical Recovery: {results['Water Recovery']['mechanical_recovery']:.2f}%")
    print(f"Total Recovery: {results['Water Recovery']['total_recovery']:.2f}%")
    
    print("\nOperating Parameters:")
    print("-" * 30)
    print(f"Gas Velocity: {results['Operating Parameters']['gas_velocity']:.2f} m/s")
    print(f"Maximum Gas Velocity: {results['Operating Parameters']['max_gas_velocity']:.2f} m/s")
    print(f"Vessel Dimensions: D = {results['Vessel']['Diameter']:.2f}m, H = {results['Vessel']['Height']:.2f}m")
    print(f"L/D Ratio: {results['Vessel']['Height']/results['Vessel']['Diameter']:.2f}")

def create_recovery_comparison_plots(base_results, opt_results):
    """Create comparison plots for recovery components and total recovery"""
    # Set style
    plt.style.use('bmh')  # Using built-in style
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Natural vs Mechanical Recovery Comparison
    configurations = ['Base Design', 'Optimized Design']
    natural_recovery = [base_results['Water Recovery']['natural_recovery'],
                       opt_results['Water Recovery']['natural_recovery']]
    mechanical_recovery = [base_results['Water Recovery']['mechanical_recovery'],
                         opt_results['Water Recovery']['mechanical_recovery']]
    
    # Calculate improvement contributions
    total_improvement = (opt_results['Water Recovery']['total_recovery'] - 
                        base_results['Water Recovery']['total_recovery'])
    natural_improvement = (opt_results['Water Recovery']['natural_recovery'] - 
                          base_results['Water Recovery']['natural_recovery'])
    mechanical_improvement = (opt_results['Water Recovery']['mechanical_recovery'] - 
                             base_results['Water Recovery']['mechanical_recovery'])
    
    natural_contribution = (natural_improvement / total_improvement) * 100
    mechanical_contribution = (mechanical_improvement / total_improvement) * 100
    
    x = np.arange(len(configurations))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, natural_recovery, width, label='Natural Recovery',
                    color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, mechanical_recovery, width, label='Mechanical Recovery',
                    color='lightgreen', alpha=0.8)
    
    ax1.set_ylabel('Recovery Percentage (%)')
    ax1.set_title('Natural vs Mechanical Recovery Comparison\n' + 
                  f'(Temperature: {natural_contribution:.1f}% of improvement\n' +
                  f'Mechanical: {mechanical_contribution:.1f}% of improvement)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configurations)
    ax1.legend()
    
    # Add value labels on bars
    for i, v in enumerate(natural_recovery):
        ax1.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(mechanical_recovery):
        ax1.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
    # Add arrows showing the changes
    # Natural recovery arrow
    mid_y_natural = (natural_recovery[0] + natural_recovery[1]) / 2
    ax1.annotate(f'+{natural_improvement:.1f}%',
                xy=(-0.17, natural_recovery[0]),
                xytext=(-0.17, natural_recovery[1]),
                ha='right',
                va='center',
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    
    # Mechanical recovery arrow
    mid_y_mech = (mechanical_recovery[0] + mechanical_recovery[1]) / 2
    ax1.annotate(f'{mechanical_improvement:.1f}%',
                xy=(0.17, mechanical_recovery[0]),
                xytext=(0.17, mechanical_recovery[1]),
                ha='left',
                va='center',
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    
    # Plot 2: Total Water Recovery Comparison
    total_recovery = [base_results['Water Recovery']['total_recovery'],
                     opt_results['Water Recovery']['total_recovery']]
    
    bars = ax2.bar(configurations, total_recovery, color=['lightcoral', 'mediumseagreen'])
    ax2.set_ylabel('Total Recovery Percentage (%)')
    ax2.set_title('Total Water Recovery Comparison')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Add improvement arrow and text
    improvement = total_recovery[1] - total_recovery[0]
    mid_y = (total_recovery[0] + total_recovery[1]) / 2
    ax2.annotate(f'+{improvement:.1f}%\nTotal Improvement',
                xy=(0.5, mid_y),
                xytext=(0.5, mid_y + 10),
                ha='center',
                va='bottom',
                arrowprops=dict(arrowstyle='->',
                              color='black',
                              lw=2))
    
    # Set y-axis limits to start from 0
    ax1.set_ylim(0, max(max(natural_recovery), max(mechanical_recovery)) * 1.2)
    ax2.set_ylim(0, max(total_recovery) * 1.1)
    
    # Add grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add temperature information
    temp_text = (f'Temperature Change: {opt_results["Operating Conditions"]["Temperature"] - base_results["Operating Conditions"]["Temperature"]:.1f}°C\n' +
                 f'({base_results["Operating Conditions"]["Temperature"]:.1f}°C → {opt_results["Operating Conditions"]["Temperature"]:.1f}°C)')
    fig.text(0.02, 0.02, temp_text, fontsize=9, style='italic')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def create_parameter_comparison_plot(base_results, opt_results):
    """Create a dot plot comparing key parameters between base and optimized designs"""
    # Set style
    plt.style.use('bmh')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define parameters and calculate percentage changes
    parameters = {
        'Temperature': {
            'base': base_results['Operating Conditions']['Temperature'] + 273.15,  # Convert to K
            'opt': opt_results['Operating Conditions']['Temperature'] + 273.15,    # Convert to K
            'unit': 'K'
        },
        'Height': {
            'base': base_results['Vessel']['Height'],
            'opt': opt_results['Vessel']['Height'],
            'unit': 'm'
        },
        'Diameter': {
            'base': base_results['Vessel']['Diameter'],
            'opt': opt_results['Vessel']['Diameter'],
            'unit': 'm'
        },
        'L/D Ratio': {
            'base': base_results['Vessel']['Height']/base_results['Vessel']['Diameter'],
            'opt': opt_results['Vessel']['Height']/opt_results['Vessel']['Diameter'],
            'unit': ''
        },
        'Gas Velocity': {
            'base': base_results['Operating Parameters']['gas_velocity'],
            'opt': opt_results['Operating Parameters']['gas_velocity'],
            'unit': 'm/s'
        },
        'Water Recovery': {
            'base': base_results['Water Recovery']['total_recovery'],
            'opt': opt_results['Water Recovery']['total_recovery'],
            'unit': '%'
        }
    }
    
    # Calculate percentage changes and sort by absolute change
    changes = []
    for param, values in parameters.items():
        pct_change = ((values['opt'] - values['base']) / abs(values['base'])) * 100
        changes.append({
            'parameter': param,
            'pct_change': pct_change,
            'base': values['base'],
            'opt': values['opt'],
            'unit': values['unit']
        })
    
    # Sort by absolute percentage change
    changes.sort(key=lambda x: abs(x['pct_change']), reverse=True)
    
    # Plot settings
    bar_height = 0.6
    y_positions = np.arange(len(changes))
    
    # Create color map based on magnitude
    max_change = max([abs(c['pct_change']) for c in changes])
    
    # Create bars
    bars = ax.barh(y_positions, [abs(c['pct_change']) for c in changes], height=bar_height)
    
    # Color bars based on magnitude
    for bar, change in zip(bars, changes):
        intensity = abs(change['pct_change']) / max_change
        bar.set_color((0, 0.5 + 0.5 * intensity, 0))  # Varying shades of green
        bar.set_alpha(0.7)
    
    # Add parameter labels and values
    for i, change in enumerate(changes):
        # Parameter name with original values
        if change['parameter'] == 'Temperature':
            # For temperature, also show Celsius in parentheses
            base_c = change['base'] - 273.15
            opt_c = change['opt'] - 273.15
            base_value = f"{change['base']:.2f}K ({base_c:.1f}°C)"
            opt_value = f"{change['opt']:.2f}K ({opt_c:.1f}°C)"
            param_text = f"{change['parameter']}\n({base_value} → {opt_value})"
        else:
            base_value = f"{change['base']:.2f}{change['unit']}"
            opt_value = f"{change['opt']:.2f}{change['unit']}"
            param_text = f"{change['parameter']}\n({base_value} → {opt_value})"
        
        ax.text(-2, i, param_text, ha='right', va='center')
        
        # Percentage change
        x_pos = abs(change['pct_change']) + 1
        direction = "decrease" if change['pct_change'] < 0 else "increase"
        ax.text(x_pos, i, f"{abs(change['pct_change']):.1f}% {direction}", 
                ha='left', va='center', color='darkgreen')
    
    # Customize plot
    ax.set_yticks([])
    ax.set_title('Magnitude of Parameter Changes (%)', pad=20)
    
    # Set x-axis limits with some padding
    ax.set_xlim(-2, max_change * 1.15)
    
    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Add explanatory text
    fig.text(0.02, 0.02, 
             "Darker green indicates larger magnitude of change\nAll changes shown as positive values",
             fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.show()

def create_stream_table(results):
    """Create a professional stream table for the optimized configuration"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    streams = {
        'Parameters': ['Pressure\n(bar)', 'Temperature\n(°C)', 'Enthalpy\nFlow (MW)', 'Mass Flow\n(kg/h)', 'Mole Flows\n(kmol h⁻¹)'],
        'Feed': [
                f"{results['Operating Conditions']['Pressure']:.2f}", 
                f"{results['Operating Conditions']['Temperature']:.1f}",
                f"{results['Energy Flows']['Feed'] / 1000:.2f}",
                f"{results['Total Mass']['Feed']:.2f}",
                f"{sum([v for v in results['Mass Flows']['Feed'].values()]) / 30:.2f}"],
        'Gas': [
                f"{results['Operating Conditions']['Pressure']:.2f}", 
                f"{results['Operating Conditions']['Temperature']:.1f}",
                f"{results['Energy Flows']['Gas Outlet'] / 1000:.2f}",
                f"{results['Total Mass']['Gas Outlet']:.2f}",
                f"{sum([v for v in results['Mass Flows']['Gas Outlet'].values()]) / 30:.2f}"],
        'Liquid': [
                f"{results['Operating Conditions']['Pressure']:.2f}", 
                f"{results['Operating Conditions']['Temperature']:.1f}",
                f"{results['Energy Flows']['Liquid Outlet'] / 1000:.2f}",
                f"{results['Total Mass']['Liquid Outlet']:.2f}",
                f"{sum([v for v in results['Mass Flows']['Liquid Outlet'].values()]) / 30:.2f}"]
    }
    
    # Create the DataFrame
    df = pd.DataFrame(streams)
    
    # Add component mole flows header
    component_header = pd.DataFrame({
        'Parameters': ['Component Mole Flows (kmol h⁻¹)'],
        'Feed': [''],
        'Gas': [''],
        'Liquid': ['']
    })
    
    # Add component rows
    components = {'CH₄': 'CH4', 'CO₂': 'CO2', 'H₂O': 'H2O', 'CO': 'CO', 'H₂': 'H2'}
    component_rows = []
    for display_name, comp_name in components.items():
        row = {
            'Parameters': display_name,
            'Feed': f"{results['Mass Flows']['Feed'].get(comp_name, 0) / PropsSI('M', comp_name):.2f}",
            'Gas': f"{results['Mass Flows']['Gas Outlet'].get(comp_name, 0) / PropsSI('M', comp_name):.2f}",
            'Liquid': f"{results['Mass Flows']['Liquid Outlet'].get(comp_name, 0) / PropsSI('M', comp_name):.2f}"
        }
        component_rows.append(row)
    
    component_df = pd.DataFrame(component_rows)
    
    # Combine all DataFrames
    final_df = pd.concat([df, component_header, component_df], ignore_index=True)
    
    # Create table
    table = ax.table(
        cellText=final_df.values,
        colLabels=['Parameters', 'Stream 5\n(Feed)', 'Stream 9\n(Gas Out)', 'Stream 8\n(Liquid Out)'],
        cellLoc='center',
        loc='center',
        colColours=['lightgray']*4
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style specific cells
    for i in range(len(final_df)):
        for j in range(len(final_df.columns)):
            cell = table[i+1, j]  # +1 because of header row
            if i < 5:  # Main stream data (updated for removed stream number)
                cell.set_text_props(weight='bold')
            elif i == 5:  # Component header
                cell.set_text_props(weight='bold', style='italic')
                cell.set_facecolor('lightgray')
    
    plt.title('Optimized KO Drum Stream Table', pad=20, weight='bold')
    plt.tight_layout()
    plt.show()

def main():
    # Create KO Drum simulator for base configuration
    base_ko_drum = KODrumPhysics()
    
    # Create KO Drum simulator for optimized configuration
    opt_ko_drum = KODrumPhysics()
    opt_ko_drum.diameter = OPT_VESSEL_DIMENSIONS['diameter']
    opt_ko_drum.height = OPT_VESSEL_DIMENSIONS['height']
    
    # Analyze both configurations with their respective operating conditions
    base_results = analyze_stream_properties(base_ko_drum, "Base", OPERATING_CONDITIONS)
    opt_results = analyze_stream_properties(opt_ko_drum, "Optimized", OPT_OPERATING_CONDITIONS)
    
    # Print results
    print_detailed_results(base_results)
    print_detailed_results(opt_results)
    
    # Calculate improvements
    total_recovery_improvement = (
        opt_results['Water Recovery']['total_recovery'] -
        base_results['Water Recovery']['total_recovery']
    )
    
    energy_balance_base = abs(base_results['Energy Flows']['Feed'] - 
                            (base_results['Energy Flows']['Gas Outlet'] + 
                             base_results['Energy Flows']['Liquid Outlet']))
    energy_balance_opt = abs(opt_results['Energy Flows']['Feed'] - 
                           (opt_results['Energy Flows']['Gas Outlet'] + 
                            opt_results['Energy Flows']['Liquid Outlet']))
    
    print("\nPerformance Comparison (Optimized vs Base):")
    print("=" * 50)
    print(f"Water Recovery Improvement: {total_recovery_improvement:.2f}%")
    print(f"Energy Balance Improvement: {energy_balance_base - energy_balance_opt:.2f} kW")
    print(f"Additional Water Recovered: {opt_results['Total Mass']['Liquid Outlet'] - base_results['Total Mass']['Liquid Outlet']:.2f} kg/h")
    print(f"Temperature Change: {opt_results['Operating Conditions']['Temperature'] - base_results['Operating Conditions']['Temperature']:.1f}°C")
    print(f"Pressure Change: {(opt_results['Operating Conditions']['Pressure'] - base_results['Operating Conditions']['Pressure'])*1000:.1f} mbar")
    
    # Create visualizations
    create_recovery_comparison_plots(base_results, opt_results)
    create_parameter_comparison_plot(base_results, opt_results)
    create_stream_table(opt_results)

if __name__ == "__main__":
    main() 