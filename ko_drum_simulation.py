import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import pandas as pd
from ko_drum_physics_properties import KODrumPhysicalProperties

class KODrum:
    def __init__(self, gas_density, liquid_density, gas_viscosity, Q_gas, Q_liquid, M_W, K=0.11, V_design=4, L_D_ratio=3):
        """
        Initialize KO Drum with design parameters
        
        Parameters:
        -----------
        gas_density : float
            Gas density in kg/m³
        liquid_density : float
            Liquid density in kg/m³
        gas_viscosity : float
            Gas viscosity in Pa.s
        Q_gas : float
            Gas volumetric flowrate in m³/s
        Q_liquid : float
            Liquid volumetric flowrate in kmol/hr
        M_W : float
            Molecular weight in kg/kmol
        K : float
            Souders-Brown constant for vertical separator (m/s)
        V_design : float
            Design gas velocity in m/s
        L_D_ratio : float
            Length to diameter ratio
        """
        self.gas_density = gas_density
        self.liquid_density = liquid_density
        self.gas_viscosity = gas_viscosity
        self.Q_gas = Q_gas
        self.Q_liquid = Q_liquid / 3600  # Convert kmol/hr to kmol/s
        self.M_W = M_W
        self.K = K
        self.V_design = V_design
        self.L_D_ratio = L_D_ratio
        
        # Calculate mass flowrate
        self.Q_mass_liquid = self.Q_liquid * self.M_W
        
        # Initialize design parameters
        self.calculate_dimensions()
        
    def peng_robinson_pressure(self, T, V, a, b, R=8.314):
        """Calculate pressure using Peng-Robinson EOS"""
        return (R * T) / (V - b) - a / (V**2 + 2 * V * b - b**2)
    
    def calculate_dimensions(self):
        """Calculate basic dimensions of the KO drum"""
        # Cross-sectional area calculation
        self.A = self.Q_gas / self.V_design
        self.D = np.sqrt((4 * self.A) / np.pi)  # Diameter
        self.L = self.D * self.L_D_ratio  # Height
        
    def calculate_max_velocity(self):
        """Calculate maximum allowable gas velocity using Souders-Brown equation"""
        self.V_max = self.K * np.sqrt((self.liquid_density - self.gas_density) / self.gas_density)
        self.V_safety = 0.8 * self.V_max  # Apply 80% safety factor
        return self.V_safety
    
    def calculate_droplet_settling(self, droplet_size=10e-6):
        """Calculate droplet settling velocity"""
        self.settling_velocity = (9.81 * droplet_size**2 * 
                                (self.liquid_density - self.gas_density)) / (18 * self.gas_viscosity)
        return self.settling_velocity
    
    def calculate_liquid_holdup(self, residence_time_minutes=5):
        """Calculate liquid holdup height for given residence time"""
        residence_time = residence_time_minutes * 60  # Convert to seconds
        self.H_liquid = (self.Q_mass_liquid / (self.A * self.liquid_density)) * residence_time
        self.H_total = self.H_liquid + self.L
        return self.H_total
    
    def plot_LD_sensitivity(self, l_d_range=(2, 5, 0.5)):
        """Plot sensitivity analysis of L/D ratio vs height"""
        L_D_values = np.arange(*l_d_range)
        heights = self.D * L_D_values
        
        plt.figure(figsize=(10, 6))
        plt.plot(L_D_values, heights, 'b-', linewidth=2, label='KO Drum Height')
        plt.xlabel('L/D Ratio')
        plt.ylabel('Height (m)')
        plt.title('Impact of L/D Ratio on KO Drum Height')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def get_design_summary(self):
        """Return a summary of the KO drum design parameters"""
        return {
            'Diameter (m)': round(self.D, 2),
            'Height (m)': round(self.H_total, 2),
            'Gas Velocity (m/s)': round(self.V_safety, 2),
            'Liquid Holdup Height (m)': round(self.H_liquid, 2),
            'L/D Ratio': self.L_D_ratio
        }

class KODrumSimulation:
    def __init__(self, operating_conditions, feed_composition, molecular_weights, vessel_dimensions):
        self.T = operating_conditions['temperature_K']
        self.P = operating_conditions['pressure_Pa']
        self.feed = feed_composition
        self.M_W = molecular_weights
        self.diameter = vessel_dimensions['diameter']
        self.height = vessel_dimensions['height']
        self.liquid_height = vessel_dimensions['liquid_holdup_height']
        
        # Initialize physics calculator
        self.physics = KODrumPhysicalProperties(self.T, self.P, self.feed, self.M_W)
        
        # Droplet properties
        self.min_droplet_size = 5e-6  # 5 microns
        self.max_droplet_size = 100e-6  # 100 microns
        self.n_droplet_sizes = 20
        self.droplet_sizes = np.linspace(self.min_droplet_size, self.max_droplet_size, self.n_droplet_sizes)
        
        # Results storage
        self.gas_outlet = {}
        self.liquid_outlet = {}
        
        # Initialize energy and mass balance
        self.energy_balance = None
        self.mass_balance_error = None
    
    def calculate_gas_velocity(self):
        """Calculate gas velocity and maximum allowable velocity"""
        # Calculate cross-sectional area
        A = np.pi * (self.diameter/2)**2
        
        # Calculate volumetric flow rate
        R = 8.314
        total_mol_s = sum(self.feed[comp] for comp in self.feed) * 1000 / 3600
        Q_gas = total_mol_s * (R * self.T / self.P)
        
        # Actual gas velocity
        v_gas = Q_gas / A
        
        # Calculate maximum velocity using Souders-Brown
        props = self.physics.calculate_fluid_properties()
        K = 0.11  # Souders-Brown coefficient
        v_max = K * np.sqrt((props['liquid_density'] - props['gas_density']) / props['gas_density'])
        
        return v_gas, v_max, Q_gas
    
    def calculate_pressure_drop(self):
        """Calculate pressure drop through the vessel"""
        v_gas, _, _ = self.calculate_gas_velocity()
        props = self.physics.calculate_fluid_properties()
        
        # Calculate Reynolds number for vessel
        Re_vessel = props['gas_density'] * v_gas * self.diameter / props['gas_viscosity']
        
        # Calculate friction factor using Colebrook correlation
        roughness = 4.5e-5  # typical roughness for commercial steel
        relative_roughness = roughness / self.diameter
        
        def colebrook(f):
            return 1/np.sqrt(f) + 2*np.log10(relative_roughness/3.7 + 2.51/(Re_vessel*np.sqrt(f)))
        
        from scipy.optimize import fsolve
        f = fsolve(colebrook, x0=0.02)[0]
        
        # Calculate pressure drop using Darcy-Weisbach equation
        dP_friction = f * (self.height/self.diameter) * (props['gas_density'] * v_gas**2) / 2
        
        # Add acceleration pressure drop
        dP_acceleration = props['gas_density'] * v_gas**2 / 2
        
        # Add gravitational pressure drop
        dP_gravity = props['gas_density'] * 9.81 * self.height
        
        total_dP = dP_friction + dP_acceleration + dP_gravity
        
        return {
            'friction_drop': dP_friction,
            'acceleration_drop': dP_acceleration,
            'gravity_drop': dP_gravity,
            'total_drop': total_dP
        }
    
    def calculate_energy_balance(self):
        """Calculate energy balance for the system"""
        # Get fluid properties
        props = self.physics.calculate_fluid_properties()
        
        # Calculate enthalpies
        H_in = sum(self.feed[comp] * self.physics.calculate_enthalpy(comp) 
                  for comp in self.feed)
        
        H_out_gas = sum(self.gas_outlet[comp] * self.physics.calculate_enthalpy(comp) 
                       for comp in self.gas_outlet)
        
        H_out_liquid = sum(self.liquid_outlet[comp] * self.physics.calculate_enthalpy(comp) 
                          for comp in self.liquid_outlet)
        
        # Calculate pressure-volume work
        v_gas, _, Q_gas = self.calculate_gas_velocity()
        dP = self.calculate_pressure_drop()['total_drop']
        PV_work = Q_gas * dP
        
        # Calculate energy balance
        energy_in = H_in
        energy_out = H_out_gas + H_out_liquid + PV_work
        
        self.energy_balance = {
            'energy_in': energy_in,
            'energy_out_gas': H_out_gas,
            'energy_out_liquid': H_out_liquid,
            'pressure_work': PV_work,
            'total_energy_out': energy_out,
            'balance_error': abs(energy_in - energy_out)
        }
        
        return self.energy_balance
    
    def verify_mass_balance(self):
        """Verify mass balance closure"""
        total_in = sum(self.feed.values())
        total_out = sum(self.gas_outlet.values()) + sum(self.liquid_outlet.values())
        
        self.mass_balance_error = abs(total_in - total_out) / total_in * 100
        
        return {
            'total_in': total_in,
            'total_out': total_out,
            'error_percent': self.mass_balance_error,
            'is_valid': self.mass_balance_error < 0.1  # 0.1% tolerance
        }
    
    def update_temperature_dependent_properties(self):
        """Update all temperature-dependent properties"""
        # Update fluid properties
        props = self.physics.calculate_fluid_properties()
        thermal_props = self.physics.calculate_thermal_properties()
        
        # Recalculate separation efficiency
        efficiency = self.calculate_separation_efficiency()
        
        # Update outlet streams
        self.calculate_outlet_streams()
        
        return {
            'fluid_properties': props,
            'thermal_properties': thermal_props,
            'separation_efficiency': efficiency
        }
    
    def calculate_separation_efficiency(self):
        """Calculate separation efficiency for current conditions"""
        # Get fluid properties
        self.physics.calculate_fluid_properties()
        
        # Get velocities and flow rates
        v_gas, v_max, Q_gas = self.calculate_gas_velocity()
        
        # Calculate efficiency for each droplet size
        efficiencies = []
        for d in self.droplet_sizes:
            droplet_physics = self.physics.calculate_droplet_physics(d, v_gas)
            v_t = droplet_physics['settling_velocity']
            
            # Calculate separation parameters
            settling_time = self.height / v_gas
            settling_distance = v_t * settling_time
            L_D_effect = np.tanh(0.5 * self.height/self.diameter)
            settling_ratio = (settling_distance / self.diameter) * L_D_effect
            
            # Calculate efficiency factors
            A_vessel = np.pi * (self.diameter/2)**2
            geometry_factor = (A_vessel * self.height) / Q_gas
            vel_factor = np.exp(-0.5 * (v_gas/v_max) * (self.height/self.diameter))
            
            # Get dimensionless numbers and effects
            numbers = self.physics.calculate_dimensionless_numbers(v_gas, d)
            coalescence_factor = 1 / (1 + 0.1 * numbers['weber'])
            
            thermal_props = self.physics.calculate_thermal_properties()
            thermal_factor = np.exp(-thermal_props['thermal_diffusivity'] * settling_time / d**2)
            
            # Combined separation parameter
            separation_parameter = (
                settling_ratio * 
                geometry_factor * 
                vel_factor * 
                coalescence_factor * 
                thermal_factor
            )
            
            # Calculate efficiency
            efficiency = 1 - np.exp(-separation_parameter)
            efficiencies.append(efficiency)
        
        # Calculate overall efficiency
        return np.mean(efficiencies)
    
    def calculate_outlet_streams(self):
        """Calculate outlet stream compositions"""
        efficiency = self.calculate_separation_efficiency()
        
        # Initialize outlet streams
        self.gas_outlet = {}
        self.liquid_outlet = {}
        
        # Calculate outlet flows
        for comp in self.feed:
            if comp == 'H2O':
                self.liquid_outlet[comp] = self.feed[comp] * efficiency
                self.gas_outlet[comp] = self.feed[comp] * (1 - efficiency)
            else:
                self.gas_outlet[comp] = self.feed[comp]
                self.liquid_outlet[comp] = 0.0
        
        return self.gas_outlet, self.liquid_outlet
    
    def simulate(self):
        """Run complete KO drum simulation with enhanced physics"""
        # Calculate separation and outlet streams
        efficiency = self.calculate_separation_efficiency()
        gas_out, liquid_out = self.calculate_outlet_streams()
        v_gas, v_max, Q_gas = self.calculate_gas_velocity()
        
        # Calculate pressure drop
        pressure_drop = self.calculate_pressure_drop()
        
        # Calculate energy balance
        energy = self.calculate_energy_balance()
        
        # Verify mass balance
        mass_balance = self.verify_mass_balance()
        
        # Get updated properties
        properties = self.update_temperature_dependent_properties()
        
        return {
            'gas_velocity': v_gas,
            'max_velocity': v_max,
            'volumetric_flow': Q_gas,
            'separation_efficiency': efficiency * 100,
            'water_recovery': efficiency * 100,
            'gas_outlet': gas_out,
            'liquid_outlet': liquid_out,
            'L_D_ratio': self.height/self.diameter,
            'pressure_drop': pressure_drop,
            'energy_balance': energy,
            'mass_balance': mass_balance,
            'properties': properties
        }

# Example usage
if __name__ == "__main__":
    # Initialize KO Drum with given parameters
    ko_drum = KODrum(
        gas_density=0.47,      # kg/m³
        liquid_density=988,    # kg/m³
        gas_viscosity=1.5e-5,  # Pa.s
        Q_gas=5.3,            # m³/s
        Q_liquid=443.12,       # kmol/hr
        M_W=18.015            # kg/kmol
    )
    
    # Calculate and print results
    max_velocity = ko_drum.calculate_max_velocity()
    settling_velocity = ko_drum.calculate_droplet_settling()
    total_height = ko_drum.calculate_liquid_holdup()
    
    # Print design summary
    summary = ko_drum.get_design_summary()
    print("\nKO Drum Design Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Plot L/D sensitivity analysis
    ko_drum.plot_LD_sensitivity() 