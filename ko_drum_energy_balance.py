import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import pandas as pd
from ko_drum_config import (
    OPERATING_CONDITIONS,
    FEED_COMPOSITION,
    MOLECULAR_WEIGHTS,
    VAPOR_FRACTION,
    VESSEL_DIMENSIONS,
    ENERGY_PARAMETERS
)

class KODrumEnergyBalance:
    def __init__(self):
        """
        Initialize KO Drum Energy Balance Calculator using configuration parameters
        """
        self.T = OPERATING_CONDITIONS['temperature_K']
        self.P = OPERATING_CONDITIONS['pressure_Pa']
        self.feed = FEED_COMPOSITION
        self.M_W = MOLECULAR_WEIGHTS
        self.vapor_fraction = VAPOR_FRACTION
        
        # Initialize results dictionaries
        self.feed_mass = {}
        self.gas_mass = {}
        self.enthalpy_data = {}
        self.calculate_enthalpies()
        
    def calculate_enthalpies(self):
        """Calculate enthalpies for each component in both phases"""
        for comp in self.feed:
            try:
                h_gas = PropsSI('H', 'T', self.T, 'P', self.P, comp)
                h_liquid = PropsSI('H', 'T', self.T, 'P', self.P, comp) if comp == "H2O" else 0
                self.enthalpy_data[comp] = {
                    "gas": h_gas,
                    "liquid": h_liquid
                }
            except Exception as e:
                print(f"Warning: Could not calculate enthalpy for {comp}. Using estimated values.")
                # Use estimated values if CoolProp fails
                self.enthalpy_data[comp] = {
                    "gas": 50000,  # Estimated value
                    "liquid": 10000 if comp == "H2O" else 0
                }
    
    def calculate_mass_balance(self):
        """Calculate mass balance for the KO drum"""
        # Calculate mass flows
        self.feed_mass = {comp: self.feed[comp] * self.M_W[comp] for comp in self.feed}
        self.total_mass_in = sum(self.feed_mass.values())
        
        # Gas outlet mass flow
        self.gas_mass = {comp: self.feed_mass[comp] * self.vapor_fraction[comp] for comp in self.feed}
        self.total_gas_out = sum(self.gas_mass.values())
        
        # Liquid outlet mass flow
        self.total_liquid_out = self.total_mass_in - self.total_gas_out
        
        # Calculate individual component flows
        self.component_mass_balance = {}
        for comp in self.feed:
            self.component_mass_balance[comp] = {
                'feed_mass': self.feed_mass[comp],
                'gas_mass': self.gas_mass[comp],
                'liquid_mass': self.feed_mass[comp] - self.gas_mass[comp]
            }
        
        return {
            "total_mass_in": self.total_mass_in,
            "total_gas_out": self.total_gas_out,
            "total_liquid_out": self.total_liquid_out,
            "component_balance": self.component_mass_balance
        }
    
    def calculate_energy_balance(self):
        """Calculate energy balance for the KO drum"""
        # Energy input
        self.enthalpy_mass = {comp: self.feed_mass[comp] * self.enthalpy_data[comp]["gas"] 
                             for comp in self.feed}
        self.total_energy_in = sum(self.enthalpy_mass.values())
        
        # Energy at gas outlet
        self.enthalpy_gas_out = {comp: self.gas_mass[comp] * self.enthalpy_data[comp]["gas"] 
                                for comp in self.feed}
        self.total_energy_gas_out = sum(self.enthalpy_gas_out.values())
        
        # Energy at liquid outlet (only water contributes)
        self.total_energy_liquid_out = (self.total_liquid_out * 
                                      self.enthalpy_data["H2O"]["liquid"])
        
        # Energy balance
        self.energy_balance = (self.total_energy_in - 
                             (self.total_energy_gas_out + self.total_energy_liquid_out))
        
        return {
            "total_energy_in": self.total_energy_in,
            "total_energy_gas_out": self.total_energy_gas_out,
            "total_energy_liquid_out": self.total_energy_liquid_out,
            "energy_balance": self.energy_balance
        }
    
    def temperature_sensitivity(self, T_range):
        """Analyze effect of temperature on energy balance"""
        energy_diffs = []
        temps = np.linspace(T_range[0], T_range[1], 20)
        
        for temp in temps:
            self.T = temp
            self.calculate_enthalpies()
            self.calculate_mass_balance()
            energy_balance = self.calculate_energy_balance()
            energy_diffs.append(energy_balance["energy_balance"])
        
        return temps, energy_diffs
    
    def plot_temperature_sensitivity(self, T_range):
        """Plot temperature sensitivity analysis"""
        temps, energy_diffs = self.temperature_sensitivity(T_range)
        
        plt.figure(figsize=(10, 6))
        plt.plot(temps - 273.15, energy_diffs, 'b-', linewidth=2)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Energy Balance Residual (J/hr)')
        plt.title('Effect of Temperature on KO Drum Energy Balance')
        plt.grid(True)
        plt.show()
    
    def get_summary(self):
        """Get summary of mass and energy balance calculations"""
        mass_balance = self.calculate_mass_balance()
        energy_balance = self.calculate_energy_balance()
        
        return pd.DataFrame({
            'Parameter': [
                'Total Mass In (kg/hr)',
                'Total Gas Out (kg/hr)',
                'Total Liquid Out (kg/hr)',
                'Total Energy In (J/hr)',
                'Total Energy Gas Out (J/hr)',
                'Total Energy Liquid Out (J/hr)',
                'Energy Balance Residual (J/hr)'
            ],
            'Value': [
                mass_balance['total_mass_in'],
                mass_balance['total_gas_out'],
                mass_balance['total_liquid_out'],
                energy_balance['total_energy_in'],
                energy_balance['total_energy_gas_out'],
                energy_balance['total_energy_liquid_out'],
                energy_balance['energy_balance']
            ]
        })

    def get_detailed_summary(self):
        """Get detailed summary of mass and energy balance calculations"""
        mass_balance = self.calculate_mass_balance()
        energy_balance = self.calculate_energy_balance()
        
        # Create component-wise summary
        component_summary = []
        for comp in self.feed:
            component_summary.append({
                'Component': comp,
                'Feed Mass (kg/hr)': mass_balance['component_balance'][comp]['feed_mass'],
                'Gas Out (kg/hr)': mass_balance['component_balance'][comp]['gas_mass'],
                'Liquid Out (kg/hr)': mass_balance['component_balance'][comp]['liquid_mass'],
                'Feed Energy (J/hr)': self.feed_mass[comp] * self.enthalpy_data[comp]['gas'],
                'Gas Energy Out (J/hr)': self.gas_mass[comp] * self.enthalpy_data[comp]['gas'],
                'Liquid Energy Out (J/hr)': (mass_balance['component_balance'][comp]['liquid_mass'] * 
                                           self.enthalpy_data[comp]['liquid'])
            })
        
        return pd.DataFrame(component_summary)

    def plot_mass_balance(self):
        """Create a visualization of the mass balance"""
        mass_balance = self.calculate_mass_balance()
        
        # Prepare data for plotting
        components = list(self.feed.keys())
        feed_masses = [mass_balance['component_balance'][comp]['feed_mass'] for comp in components]
        gas_masses = [mass_balance['component_balance'][comp]['gas_mass'] for comp in components]
        liquid_masses = [mass_balance['component_balance'][comp]['liquid_mass'] for comp in components]
        
        # Create bar positions
        x = np.arange(len(components))
        width = 0.25
        
        # Create grouped bar chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Component-wise distribution
        rects1 = ax1.bar(x - width, feed_masses, width, label='Feed')
        rects2 = ax1.bar(x, gas_masses, width, label='Gas Out')
        rects3 = ax1.bar(x + width, liquid_masses, width, label='Liquid Out')
        
        ax1.set_ylabel('Mass Flow Rate (kg/hr)')
        ax1.set_title('Component-wise Mass Balance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(components)
        ax1.legend()
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90)
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        # Overall mass balance pie chart
        total_in = mass_balance['total_mass_in']
        total_gas = mass_balance['total_gas_out']
        total_liquid = mass_balance['total_liquid_out']
        
        labels = ['Gas Out', 'Liquid Out']
        sizes = [total_gas, total_liquid]
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                startangle=90)
        ax2.axis('equal')
        ax2.set_title('Overall Mass Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def get_mass_balance_report(self):
        """Generate a detailed mass balance report"""
        mass_balance = self.calculate_mass_balance()
        
        # Calculate component percentages
        total_mass = mass_balance['total_mass_in']
        component_data = []
        
        for comp in self.feed:
            comp_data = mass_balance['component_balance'][comp]
            feed_mass = comp_data['feed_mass']
            gas_mass = comp_data['gas_mass']
            liquid_mass = comp_data['liquid_mass']
            
            component_data.append({
                'Component': comp,
                'Feed Mass (kg/hr)': feed_mass,
                'Feed Mass %': (feed_mass/total_mass) * 100,
                'Gas Out (kg/hr)': gas_mass,
                'Gas Out %': (gas_mass/feed_mass) * 100 if feed_mass > 0 else 0,
                'Liquid Out (kg/hr)': liquid_mass,
                'Liquid Out %': (liquid_mass/feed_mass) * 100 if feed_mass > 0 else 0,
                'Mass Balance Error (kg/hr)': feed_mass - (gas_mass + liquid_mass)
            })
        
        return pd.DataFrame(component_data)

# Example usage
if __name__ == "__main__":
    # Create KO Drum simulator
    ko_drum = KODrumEnergyBalance()
    
    # Get and print mass balance report
    mass_report = ko_drum.get_mass_balance_report()
    print("\nDetailed Mass Balance Report:")
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(mass_report.to_string(index=False))
    
    # Plot mass balance visualization
    print("\nGenerating mass balance visualizations...")
    ko_drum.plot_mass_balance()
    
    # Get and print summary
    summary = ko_drum.get_summary()
    print("\nKO Drum Overall Mass and Energy Balance Summary:")
    print(summary.to_string(index=False))
    
    # Get and print detailed component summary
    detailed_summary = ko_drum.get_detailed_summary()
    print("\nKO Drum Component-wise Mass and Energy Balance:")
    print(detailed_summary.to_string(index=False))
    
    # Plot temperature sensitivity
    print("\nGenerating temperature sensitivity plot...")
    ko_drum.plot_temperature_sensitivity((30 + 273.15, 70 + 273.15)) 