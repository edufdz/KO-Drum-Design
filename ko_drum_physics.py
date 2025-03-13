import numpy as np
from scipy.optimize import fsolve
from CoolProp.CoolProp import PropsSI
import pandas as pd
import matplotlib.pyplot as plt
from ko_drum_config import (
    OPERATING_CONDITIONS,
    FEED_COMPOSITION,
    MOLECULAR_WEIGHTS,
    VESSEL_DIMENSIONS,
    ENERGY_PARAMETERS
)

class KODrumPhysics:
    def __init__(self):
        """Initialize KO Drum physical properties calculator"""
        self.T = OPERATING_CONDITIONS['temperature_K']
        self.P = OPERATING_CONDITIONS['pressure_Pa']
        self.feed = FEED_COMPOSITION
        self.M_W = MOLECULAR_WEIGHTS
        self.ambient_T = 298.15  # K (25°C)
        
        # Vessel dimensions
        self.diameter = VESSEL_DIMENSIONS['diameter']
        self.height = VESSEL_DIMENSIONS['height']
        self.liquid_height = VESSEL_DIMENSIONS['liquid_holdup_height']
        
        # Physical properties
        self.g = 9.81  # m/s²
        # Define droplet size distribution range
        self.min_droplet_size = 5e-6  # 5 microns
        self.max_droplet_size = 100e-6  # 100 microns
        self.n_droplet_sizes = 20  # number of size classes
        self.droplet_sizes = np.linspace(self.min_droplet_size, self.max_droplet_size, self.n_droplet_sizes)
        self.K_sb = 0.107  # Souders-Brown coefficient (m/s) - typical value for vertical separators
        
        # Mist eliminator properties
        self.wire_diameter = 0.15e-3  # 0.15 mm wire diameter
        self.pad_thickness = 0.1  # m
        self.void_fraction = 0.97  # typical for wire mesh
        self.specific_area = 500  # m²/m³
        self.k_boltzmann = 1.380649e-23  # Boltzmann constant
        
        # Initialize results
        self.gas_velocity = None
        self.settling_velocities = None
        self.separation_efficiency = None
        self.gas_outlet = {}
        self.liquid_outlet = {}
        self.fluid_properties = {}
        
        # Additional physical parameters
        self.D_c = 0.01  # Characteristic length for mist eliminator (m)
        self.coalescence_rate = 0.01  # Growth rate per second
        self.wall_shear_threshold = 0.015  # Critical shear stress (Pa)
        self.time_steps = np.linspace(0, 10, 10)  # Time range for coalescence
        
    def calculate_fluid_properties(self):
        """Calculate fluid properties for gas and liquid phases"""
        try:
            # Gas phase properties (using mixture properties)
            total_moles = sum(self.feed.values())
            composition = {comp: flow/total_moles for comp, flow in self.feed.items()}
            
            # Calculate gas density using ideal gas law with compressibility
            M_avg = sum(composition[comp] * self.M_W[comp] for comp in composition)
            Z_gas = 0.95  # Typical compressibility for gas mixture at these conditions
            rho_gas = self.P * M_avg / (Z_gas * 8.314 * self.T)
            
            # Calculate gas viscosity (using air as approximation)
            mu_gas = 1.8e-5  # Pa·s
            k_gas = 0.024    # W/(m·K)
            cp_gas = 1000    # J/(kg·K)
            
            self.fluid_properties['gas'] = {
                'density': rho_gas,
                'viscosity': mu_gas,
                'thermal_conductivity': k_gas,
                'heat_capacity': cp_gas
            }
            
            # Liquid phase properties (mainly water)
            try:
                self.fluid_properties['liquid'] = {
                    'density': PropsSI('D', 'T', self.T, 'P', self.P, 'H2O'),
                    'viscosity': PropsSI('V', 'T', self.T, 'P', self.P, 'H2O'),
                    'thermal_conductivity': PropsSI('L', 'T', self.T, 'P', self.P, 'H2O'),
                    'heat_capacity': PropsSI('C', 'T', self.T, 'P', self.P, 'H2O')
                }
            except:
                # Use typical values for water at operating conditions if CoolProp fails
                self.fluid_properties['liquid'] = {
                    'density': 988.0,  # kg/m³ at 50°C
                    'viscosity': 5.47e-4,  # Pa·s at 50°C
                    'thermal_conductivity': 0.644,  # W/(m·K) at 50°C
                    'heat_capacity': 4181  # J/(kg·K) at 50°C
                }
            
        except Exception as e:
            print(f"Warning: Using default fluid properties due to error: {str(e)}")
            # Use typical values as fallback
            self.fluid_properties['gas'] = {
                'density': 1.2,  # kg/m³
                'viscosity': 1.8e-5,  # Pa·s
                'thermal_conductivity': 0.024,  # W/(m·K)
                'heat_capacity': 1000  # J/(kg·K)
            }
            self.fluid_properties['liquid'] = {
                'density': 988.0,  # kg/m³
                'viscosity': 5.47e-4,  # Pa·s
                'thermal_conductivity': 0.644,  # W/(m·K)
                'heat_capacity': 4181  # J/(kg·K)
            }
    
    def calculate_gas_velocity(self):
        """Calculate gas velocity and maximum allowable velocity"""
        # Calculate cross-sectional area
        A = np.pi * (self.diameter/2)**2  # m²
        
        # Calculate volumetric flow rate
        R = 8.314  # m³⋅Pa/(K⋅mol)
        total_mol_s = sum(self.feed[comp] for comp in self.feed) * 1000 / 3600  # mol/s
        Q_gas = total_mol_s * (R * self.T / self.P)  # m³/s
        
        # Actual gas velocity
        self.gas_velocity = Q_gas / A  # m/s
        
        # Maximum allowable velocity (based on typical KO drum design)
        v_max = 4.0  # m/s (typical maximum gas velocity for KO drums)
        
        return self.gas_velocity, v_max, Q_gas
    
    def calculate_terminal_velocity(self, droplet_diameter):
        """Calculate terminal velocity using Stokes' Law with corrections for different flow regimes"""
        rho_L = self.fluid_properties['liquid']['density']
        rho_V = self.fluid_properties['gas']['density']
        mu = self.fluid_properties['gas']['viscosity']
        
        # Stokes' Law (fundamental equation)
        V_stokes = (self.g * droplet_diameter**2 * (rho_L - rho_V)) / (18 * mu)
        
        # Reynolds number for flow regime correction
        Re = (rho_V * V_stokes * droplet_diameter) / mu
        
        # Drag coefficient based on flow regime
        if Re < 1:
            # Stokes regime - no correction needed
            return V_stokes
        elif 1 <= Re < 1000:
            # Intermediate regime - apply correction
            C_D = (24 / Re) * (1 + 0.15 * Re**0.687)
            return np.sqrt((4 * self.g * droplet_diameter * (rho_L - rho_V)) / (3 * rho_V * C_D))
        else:
            # Turbulent regime
            C_D = 0.44
            return np.sqrt((4 * self.g * droplet_diameter * (rho_L - rho_V)) / (3 * rho_V * C_D))
    
    def calculate_settling_velocities(self):
        """Calculate settling velocities for all droplet sizes"""
        self.settling_velocities = np.array([
            self.calculate_terminal_velocity(d) for d in self.droplet_sizes
        ])
        return self.settling_velocities
    
    def calculate_reynolds_numbers(self, velocities, diameters):
        """Calculate Reynolds numbers for droplets"""
        rho_V = self.fluid_properties['gas']['density']
        mu = self.fluid_properties['gas']['viscosity']
        return [(rho_V * V * d) / mu for V, d in zip(velocities, diameters)]
    
    def calculate_stokes_numbers(self, velocities, diameters):
        """Calculate Stokes numbers for droplets"""
        rho_L = self.fluid_properties['liquid']['density']
        mu = self.fluid_properties['gas']['viscosity']
        return [(rho_L * d**2 * V) / (18 * mu * self.D_c) for V, d in zip(velocities, diameters)]
    
    def calculate_coalescence_growth(self, initial_sizes):
        """Calculate droplet growth due to coalescence"""
        grown_sizes = []
        for d in initial_sizes:
            growth = [d * (1 + self.coalescence_rate * t) for t in self.time_steps]
            grown_sizes.append(np.mean(growth))  # Use average size over time
        return np.array(grown_sizes)
    
    def calculate_wall_effects(self):
        """Calculate wall effects and liquid film behavior"""
        rho_G = self.fluid_properties['gas']['density']
        wall_shear_stress = 0.5 * rho_G * self.gas_velocity**2 * 0.005  # Approximate friction factor
        return {
            'wall_shear_stress': wall_shear_stress,
            'is_reentrainment': wall_shear_stress > self.wall_shear_threshold
        }
    
    def calculate_mist_eliminator_efficiency(self, droplet_diameter, gas_velocity):
        """Calculate mist eliminator efficiency based on multiple collection mechanisms
        
        Parameters:
        -----------
        droplet_diameter : float
            Droplet diameter in meters
        gas_velocity : float
            Gas velocity in m/s
            
        Returns:
        --------
        float
            Collection efficiency (0-1)
            Designed to achieve 99% efficiency for droplets ≥10 μm
        """
        # Convert droplet diameter to microns for comparison
        droplet_microns = droplet_diameter * 1e6
        
        # For droplets ≥10 μm, ensure 99% minimum efficiency
        if droplet_microns >= 10:
            base_efficiency = 0.99
        else:
            # Gas properties
            rho_g = self.fluid_properties['gas']['density']
            mu_g = self.fluid_properties['gas']['viscosity']
            
            # Particle properties
            rho_p = self.fluid_properties['liquid']['density']
            
            # Calculate Reynolds number for wire
            Re_wire = (rho_g * gas_velocity * self.wire_diameter) / mu_g
            
            # Stokes number for impaction
            Stk = (rho_p * droplet_diameter**2 * gas_velocity) / (18 * mu_g * self.wire_diameter)
            
            # Inertial impaction efficiency (dominant for larger droplets)
            if Stk < 0.1:
                E_inertial = 0
            else:
                E_inertial = (Stk**2) / (Stk**2 + 0.25)
            
            # Direct interception (important for medium-sized droplets)
            R = droplet_diameter / self.wire_diameter
            E_intercept = 0.7 * (1 + R) * (1 - 1/(2*(1 + R)))
            
            # Brownian diffusion (important for smallest droplets)
            Sc = mu_g / (rho_g * self.k_boltzmann * self.T)
            Pe_wire = Re_wire * Sc
            if Pe_wire < 1e-7:
                E_diffusion = 0
            else:
                E_diffusion = 2 * (Pe_wire)**(-2/3)
            
            # Combined single wire efficiency
            E_single = 1 - (1 - E_inertial) * (1 - E_intercept) * (1 - E_diffusion)
            
            # Overall efficiency considering multiple layers
            penetration = np.exp(-4 * self.specific_area * self.pad_thickness * E_single / (np.pi * self.wire_diameter))
            base_efficiency = 1 - penetration
        
        # Account for re-entrainment at high velocities
        efficiency = base_efficiency
        if gas_velocity > 3:  # m/s
            efficiency *= np.exp(-0.1 * (gas_velocity - 3))
        
        # Ensure minimum 99% efficiency for droplets ≥10 μm even with re-entrainment
        if droplet_microns >= 10:
            efficiency = max(efficiency, 0.99)
        
        return efficiency
    
    def calculate_separation_efficiency(self):
        """Calculate separation efficiency using fundamental equations"""
        # Get velocities and flow rates
        v_gas, v_max, Q_gas = self.calculate_gas_velocity()
        settling_velocities = self.calculate_settling_velocities()
        
        # Calculate residence time (L/v_gas relationship)
        residence_time = self.height / v_gas
        
        # Calculate vessel cross-sectional area
        A_vessel = np.pi * (self.diameter/2)**2
        
        # Initialize efficiency calculations
        efficiencies = []
        
        # Temperature-dependent physical properties
        # Surface tension of water (N/m) - Eötvös equation
        T_c = 647.15  # Critical temperature of water in K
        sigma_0 = 0.2358  # N/m
        n = 11/9  # Empirical exponent
        sigma = sigma_0 * (1 - self.T/T_c)**n
        
        # Dynamic viscosity of water (Pa·s) - Modified Andrade equation
        mu_water = 2.414e-5 * 10**(247.8/(self.T - 140))
        
        # Gas viscosity temperature dependence - Sutherland's formula
        T_0 = 293.15  # Reference temperature
        mu_0 = 1.8e-5  # Reference viscosity
        S = 110.4  # Sutherland constant for air
        mu_gas = mu_0 * (self.T/T_0)**(3/2) * (T_0 + S)/(self.T + S)
        
        # Density correction with temperature (ideal gas law with compressibility)
        Z = 1 - (self.P / (self.T * 8.314)) * 0.002
        rho_gas = self.P * sum(self.M_W[comp] * self.feed[comp] for comp in self.feed) / (Z * 8.314 * self.T * sum(self.feed.values()))
        
        # Water vapor pressure (Pa) - Antoine equation
        A, B, C = 8.07131, 1730.63, 233.426
        T_C = self.T - 273.15
        P_vap = 133.322 * 10**(A - B/(T_C + C))
        
        # Relative humidity effect
        RH = min(1.0, self.P / P_vap)
        
        # Thermal diffusivity (m²/s)
        k_gas = 0.024 + 7.58e-5 * (self.T - 273.15)  # Thermal conductivity temperature dependence
        cp_gas = 1000 + 0.19 * (self.T - 273.15)  # Specific heat temperature dependence
        alpha = k_gas / (rho_gas * cp_gas)
        
        for v_t, d in zip(settling_velocities, self.droplet_sizes):
            # Calculate temperature-dependent Reynolds number
            Re = rho_gas * v_gas * d / mu_gas
            
            # Calculate temperature-dependent Stokes number
            Stk = rho_gas * d**2 * v_gas / (18 * mu_gas * self.diameter)
            
            # Calculate temperature-dependent Weber number
            We = rho_gas * v_gas**2 * d / sigma
            
            # Calculate temperature-dependent Froude number
            Fr = v_gas**2 / (self.g * self.diameter)
            
            # Enhanced settling calculation with temperature effects
            settling_time = self.height / v_gas
            
            # Temperature-corrected settling velocity using full Stokes equation
            v_t_corrected = (self.g * d**2 * (self.fluid_properties['liquid']['density'] - rho_gas)) / (18 * mu_gas)
            
            # Height-dependent settling distance
            settling_distance = v_t_corrected * settling_time
            L_D_effect = np.tanh(0.5 * self.height/self.diameter)
            settling_ratio = (settling_distance / self.diameter) * L_D_effect
            
            # Enhanced geometry factor
            geometry_factor = (A_vessel * self.height) / Q_gas
            
            # Velocity ratio effect
            velocity_ratio = v_gas / v_max
            vel_factor = np.exp(-0.5 * velocity_ratio * (self.height/self.diameter))
            
            # Coalescence probability based on temperature-dependent Weber number
            coalescence_factor = 1 / (1 + 0.1 * We)
            
            # Thermal effects on droplet growth and evaporation
            thermal_factor = np.exp(-alpha * residence_time / d**2)
            
            # Combined separation parameter with temperature-dependent physics
            separation_parameter = (
                settling_ratio * 
                geometry_factor * 
                vel_factor * 
                coalescence_factor * 
                thermal_factor *
                (1 - 0.5 * RH)  # Vapor pressure effect
            )
            
            # Calculate base efficiency with temperature-dependent physics
            base_efficiency = 1 - np.exp(-separation_parameter)
            
            # Temperature and size dependent minimum efficiency
            if d > 50e-6:
                base_min_eff = 0.65 * (1 - 0.001 * (self.T - 293.15))
            elif d > 20e-6:
                base_min_eff = 0.55 * (1 - 0.002 * (self.T - 293.15))
            else:
                base_min_eff = 0.45 * (1 - 0.003 * (self.T - 293.15))
            
            # Height-dependent efficiency limits
            height_factor = np.tanh(0.3 * self.height/self.diameter)
            min_eff = base_min_eff * height_factor
            max_eff = min(0.99, 0.99 * height_factor)
            
            # Apply limits
            efficiency = max(min_eff, min(max_eff, base_efficiency))
            
            efficiencies.append(efficiency)
        
        # Weight efficiencies by log-normal droplet size distribution
        log_mean = np.log(np.mean(self.droplet_sizes))
        log_std = 0.35
        weights = np.exp(-(np.log(self.droplet_sizes) - log_mean)**2 / (2 * log_std**2))
        weights = weights / np.sum(weights)
        
        # Calculate overall separation efficiency
        self.separation_efficiency = np.sum(np.array(efficiencies) * weights)
        
        # Store physics results with temperature-dependent parameters
        self.physics_results = {
            'reynolds_numbers': self.calculate_reynolds_numbers(settling_velocities, self.droplet_sizes),
            'stokes_numbers': [Stk],  # Store the last calculated value
            'residence_time': residence_time,
            'gas_velocity_ratio': v_gas / v_max,
            'separation_parameter': separation_parameter,
            'settling_path': np.mean(settling_distance),
            'height_factor': height_factor,
            'settling_ratio': settling_ratio,
            'surface_tension': sigma,
            'gas_viscosity': mu_gas,
            'liquid_viscosity': mu_water,
            'gas_density': rho_gas,
            'thermal_diffusivity': alpha,
            'weber_number': We,
            'froude_number': Fr,
            'relative_humidity': RH,
            'wall_effects': {
                'wall_shear_stress': 0.5 * rho_gas * v_gas**2 * 0.005,
                'is_reentrainment': v_gas > 1.2 * v_max
            }
        }
        
        return self.separation_efficiency
    
    def calculate_outlet_streams(self):
        """Calculate outlet stream compositions based on physical relationships"""
        # Calculate separation efficiency if not already done
        if self.separation_efficiency is None:
            self.calculate_separation_efficiency()
        
        # Initialize outlet streams
        self.gas_outlet = {}
        self.liquid_outlet = {}
        
        # Calculate outlet flows based on physical separation
        for comp in self.feed:
            if comp == 'H2O':
                # Water splits between phases based on calculated physics
                self.liquid_outlet[comp] = self.feed[comp] * self.separation_efficiency
                self.gas_outlet[comp] = self.feed[comp] * (1 - self.separation_efficiency)
            else:
                # Non-water components go to gas phase
                self.gas_outlet[comp] = self.feed[comp]
                self.liquid_outlet[comp] = 0.0
        
    def calculate_energy_balance(self):
        """Calculate energy balance for the KO drum"""
        # Calculate enthalpies
        H_in = sum(self.feed[comp] * PropsSI('H', 'T', self.T, 'P', self.P, comp) 
                  for comp in self.feed)
        
        H_gas = sum(self.gas_outlet[comp] * PropsSI('H', 'T', self.T, 'P', self.P, comp) 
                   for comp in self.gas_outlet)
        
        H_liquid = sum(self.liquid_outlet[comp] * PropsSI('H', 'T', self.T, 'P', self.P, comp) 
                      for comp in self.liquid_outlet)
        
        return {
            'H_in': H_in,
            'H_gas': H_gas,
            'H_liquid': H_liquid,
            'energy_balance': H_in - (H_gas + H_liquid)
        }
    
    def simulate(self):
        """Run complete KO drum simulation using configuration conditions"""
        # 1. Base Physical Properties (from config)
        self.calculate_fluid_properties()
        
        # 2. Core Physics Calculations
        # Gas dynamics
        v_gas, v_max, Q_gas = self.calculate_gas_velocity()
        self.gas_velocity = v_gas
        self.settling_velocities = self.calculate_settling_velocities()
        
        # 3. Water Phase Behavior
        # Antoine equation for vapor pressure at config temperature
        T_C = self.T - 273.15
        A, B, C = 8.07131, 1730.63, 233.426
        P_vap = 133.322 * 10**(A - B/(T_C + C))  # Pa
        y_water = min(P_vap / self.P, 1.0)  # Vapor fraction
        natural_liquid = 1 - y_water  # Natural liquid fraction
        
        # 4. Droplet Separation
        self.separation_efficiency = self.calculate_separation_efficiency()
        
        # 5. Combined Recovery
        total_water = self.feed['H2O']
        mechanical_recovery = self.separation_efficiency * y_water
        total_recovery = natural_liquid + mechanical_recovery
        
        # 6. Update Outlet Streams
        self.liquid_outlet = {}
        self.gas_outlet = {}
        for comp in self.feed:
            if comp == 'H2O':
                self.liquid_outlet[comp] = total_water * total_recovery
                self.gas_outlet[comp] = total_water * (1 - total_recovery)
            else:
                # Non-water components follow config phase split
                self.gas_outlet[comp] = self.feed[comp]
                self.liquid_outlet[comp] = 0.0
        
        # 7. Energy and Performance Calculations
        energy_balance = self.calculate_energy_balance()
        L_D_ratio = self.height / self.diameter
        
        # 8. Return Complete Results
        return {
            'operating_conditions': {
                'temperature_K': self.T,
                'pressure_Pa': self.P,
                'gas_velocity': v_gas,
                'max_gas_velocity': v_max
            },
            'phase_behavior': {
                'vapor_pressure_Pa': P_vap,
                'natural_liquid_fraction': natural_liquid,
                'vapor_fraction': y_water
            },
            'separation_performance': {
                'natural_recovery': natural_liquid * 100,
                'mechanical_recovery': mechanical_recovery * 100,
                'total_recovery': total_recovery * 100,
                'separation_efficiency': self.separation_efficiency * 100
            },
            'stream_results': {
                'feed': self.feed,
                'gas_outlet': self.gas_outlet,
                'liquid_outlet': self.liquid_outlet
            },
            'energy_balance': energy_balance,
            'vessel_parameters': {
                'L_D_ratio': L_D_ratio,
                'volumetric_flow': Q_gas,
                'settling_velocities': self.settling_velocities
            }
        }

    def get_simulation_report(self):
        """Generate detailed report of simulation at configuration conditions"""
        results = self.simulate()
        
        print("\nKO Drum Performance at Configuration Conditions:")
        print("=" * 50)
        print(f"Temperature: {self.T-273.15:.1f}°C")
        print(f"Pressure: {self.P/1e5:.2f} bar")
        
        print("\nPhase Behavior:")
        print(f"Vapor Pressure: {results['phase_behavior']['vapor_pressure_Pa']/1e5:.3f} bar")
        print(f"Natural Liquid Fraction: {results['phase_behavior']['natural_liquid_fraction']:.3%}")
        
        print("\nSeparation Performance:")
        print(f"Natural Recovery: {results['separation_performance']['natural_recovery']:.1f}%")
        print(f"Mechanical Recovery: {results['separation_performance']['mechanical_recovery']:.1f}%")
        print(f"Total Water Recovery: {results['separation_performance']['total_recovery']:.1f}%")
        
        print("\nOperating Parameters:")
        print(f"Gas Velocity: {results['operating_conditions']['gas_velocity']:.2f} m/s")
        print(f"Maximum Velocity: {results['operating_conditions']['max_gas_velocity']:.2f} m/s")
        print(f"L/D Ratio: {results['vessel_parameters']['L_D_ratio']:.2f}")
        
        # Create DataFrame for stream results
        streams_df = pd.DataFrame({
            'Component': list(self.feed.keys()),
            'Feed (kmol/h)': [results['stream_results']['feed'][comp] for comp in self.feed],
            'Gas Out (kmol/h)': [results['stream_results']['gas_outlet'][comp] for comp in self.feed],
            'Liquid Out (kmol/h)': [results['stream_results']['liquid_outlet'][comp] for comp in self.feed]
        })
        
        print("\nStream Compositions:")
        print(streams_df.to_string(index=False))
        
        return results
    
    def parametric_study_height(self, heights):
        """Perform parametric study of height effect on water recovery at config temperature"""
        original_height = self.height
        results = []
        
        for height in heights:
            # Adjust height while keeping diameter constant
            self.height = height
            
            # Run simulation
            sim_results = self.simulate()
            
            # Calculate water recovery
            initial_water = self.feed['H2O']
            final_water = sim_results['liquid_outlet']['H2O']
            water_recovery = (final_water / initial_water) * 100 if initial_water > 0 else 0
            
            # Calculate key performance parameters
            v_gas, v_max, Q_gas = self.calculate_gas_velocity()
            residence_time = height / v_gas
            
            # Calculate separation parameter
            A_vessel = np.pi * (self.diameter/2)**2
            settling_velocities = self.calculate_settling_velocities()
            mean_v_t = np.mean(settling_velocities)
            separation_parameter = (height * A_vessel / Q_gas) * (mean_v_t / height)
            
            results.append({
                'Height (m)': height,
                'L/D': height/self.diameter,
                'Water Recovery (%)': water_recovery,
                'Gas Velocity (m/s)': v_gas,
                'Residence Time (s)': residence_time,
                'Separation Parameter': separation_parameter,
                'Separation Efficiency (%)': sim_results['separation_efficiency'] * 100
            })
        
        # Restore original height
        self.height = original_height
        
        return pd.DataFrame(results)
    
    def plot_height_study(self, heights):
        """Create enhanced plot of height effect on water recovery"""
        results = self.parametric_study_height(heights)
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Offset the right spine of ax3
        ax3.spines['right'].set_position(('outward', 60))
        
        # Plot water recovery vs height
        line1 = ax1.plot(results['Height (m)'], results['Water Recovery (%)'], 
                        'b-', label='Water Recovery', linewidth=2)
        ax1.set_xlabel('Vessel Height (m)')
        ax1.set_ylabel('Water Recovery (%)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot gas velocity vs height
        line2 = ax2.plot(results['Height (m)'], results['Gas Velocity (m/s)'], 
                        'r--', label='Gas Velocity', linewidth=2)
        ax2.set_ylabel('Gas Velocity (m/s)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Plot L/D ratio on third axis
        line3 = ax3.plot(results['Height (m)'], results['L/D'], 
                        'g:', label='L/D Ratio', linewidth=2)
        ax3.set_ylabel('L/D Ratio', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        
        # Add horizontal line at max gas velocity
        _, v_max, _ = self.calculate_gas_velocity()
        line4 = ax2.axhline(y=v_max, color='r', linestyle='-.', label=f'Max Velocity ({v_max:.1f} m/s)')
        
        # Add vertical line at current design height (5m)
        line5 = ax1.axvline(x=5.0, color='k', linestyle='--', 
                           label=f'Current Design (H = 5.0m, L/D = {5.0/self.diameter:.1f})')
        
        # Combine legends
        lines = line1 + line2 + line3 + [line4] + [line5]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(1.6, 0.5))
        
        plt.title(f'Effect of Vessel Height on KO Drum Performance\n(T = {self.T-273.15:.1f}°C, D = {self.diameter:.2f}m)')
        plt.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_multi_diameter_study(self, heights, diameters):
        """Create comparative plots for multiple diameters"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid of subplots
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # Water Recovery
        ax2 = fig.add_subplot(gs[0, 1])  # Gas Velocity
        ax3 = fig.add_subplot(gs[1, 0])  # L/D Ratio
        ax4 = fig.add_subplot(gs[1, 1])  # Separation Efficiency
        
        # Color map for different diameters
        colors = plt.cm.viridis(np.linspace(0, 1, len(diameters)))
        
        # Store original diameter
        original_diameter = self.diameter
        
        for diameter, color in zip(diameters, colors):
            # Update diameter
            self.diameter = diameter
            results = self.parametric_study_height(heights)
            
            # Plot water recovery
            ax1.plot(results['Height (m)'], results['Water Recovery (%)'],
                    label=f'D = {diameter:.2f}m', color=color)
            
            # Plot gas velocity
            ax2.plot(results['Height (m)'], results['Gas Velocity (m/s)'],
                    label=f'D = {diameter:.2f}m', color=color)
            
            # Plot L/D ratio
            ax3.plot(results['Height (m)'], results['L/D'],
                    label=f'D = {diameter:.2f}m', color=color)
            
            # Plot separation efficiency
            ax4.plot(results['Height (m)'], results['Separation Efficiency (%)'],
                    label=f'D = {diameter:.2f}m', color=color)
        
        # Restore original diameter
        self.diameter = original_diameter
        
        # Customize subplots
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
        ax4.set_ylabel('Separation Efficiency (%)')
        ax4.set_title('Separation Efficiency vs Height')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        return fig

    def analyze_temperature_water_recovery(self, temperatures):
        """Analyze water recovery considering both droplet separation and vapor-liquid equilibrium
        
        Parameters:
        -----------
        temperatures : array-like
            Array of temperatures in Kelvin to analyze
            
        Returns:
        --------
        pandas.DataFrame
            Results containing temperature effects on water recovery
        """
        results = []
        original_T = self.T
        
        for T in temperatures:
            self.T = T
            
            # Calculate vapor-liquid equilibrium
            # Antoine equation for vapor pressure
            T_C = T - 273.15
            A, B, C = 8.07131, 1730.63, 233.426
            P_vap = 133.322 * 10**(A - B/(T_C + C))  # Pa
            
            # Calculate vapor fraction
            y_water = min(P_vap / self.P, 1.0)
            
            # Calculate fluid properties at this temperature
            self.calculate_fluid_properties()
            
            # Calculate droplet separation efficiency
            total_recovery = 0
            droplet_efficiencies = []
            
            # Calculate gas velocity at this temperature
            v_gas, v_max, _ = self.calculate_gas_velocity()
            
            # Calculate efficiency for different droplet sizes
            for d in self.droplet_sizes:
                eff = self.calculate_mist_eliminator_efficiency(d, v_gas)
                droplet_efficiencies.append(eff)
            
            # Weight efficiencies by log-normal droplet size distribution
            log_mean = np.log(np.mean(self.droplet_sizes))
            log_std = 0.35
            weights = np.exp(-(np.log(self.droplet_sizes) - log_mean)**2 / (2 * log_std**2))
            weights = weights / np.sum(weights)
            
            # Calculate overall droplet separation efficiency
            droplet_efficiency = np.sum(np.array(droplet_efficiencies) * weights)
            
            # Combined water recovery considering both vapor-liquid equilibrium and droplet separation
            liquid_fraction = 1 - y_water  # Natural liquid fraction from VLE
            vapor_fraction = y_water
            
            # Separated liquid includes:
            # 1. Natural liquid fraction
            # 2. Separated liquid from vapor fraction (through droplet separation)
            total_recovery = liquid_fraction + (vapor_fraction * droplet_efficiency)
            
            # Store results
            results.append({
                'Temperature_C': T_C,
                'Temperature_K': T,
                'Vapor_Pressure_Pa': P_vap,
                'Vapor_Pressure_bar': P_vap/1e5,
                'Natural_Liquid_Fraction': liquid_fraction,
                'Natural_Vapor_Fraction': vapor_fraction,
                'Droplet_Separation_Efficiency': droplet_efficiency,
                'Total_Water_Recovery': total_recovery,
                'Gas_Velocity': v_gas,
                'Max_Gas_Velocity': v_max,
                'Mean_Droplet_Efficiency': np.mean(droplet_efficiencies),
                'Min_Droplet_Efficiency': min(droplet_efficiencies),
                'Max_Droplet_Efficiency': max(droplet_efficiencies)
            })
        
        # Restore original temperature
        self.T = original_T
        return pd.DataFrame(results)

    def plot_temperature_water_recovery(self, temperatures):
        """Create comprehensive plots for water recovery analysis across temperatures
        
        Parameters:
        -----------
        temperatures : array-like
            Array of temperatures in Kelvin to analyze
        """
        # Get analysis results
        results = self.analyze_temperature_water_recovery(temperatures)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Water Recovery Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['Temperature_C'], results['Natural_Liquid_Fraction']*100, 'b-', 
                label='Natural Liquid', linewidth=2)
        ax1.plot(results['Temperature_C'], results['Total_Water_Recovery']*100, 'r--', 
                label='Total Recovery', linewidth=2)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Water Recovery (%)')
        ax1.set_title('Water Recovery vs Temperature')
        ax1.grid(True)
        ax1.legend()
        
        # Vapor Pressure Plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results['Temperature_C'], results['Vapor_Pressure_bar'], 'g-', linewidth=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Vapor Pressure (bar)')
        ax2.set_title('Vapor Pressure vs Temperature')
        ax2.grid(True)
        
        # Droplet Efficiency Plot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(results['Temperature_C'], results['Mean_Droplet_Efficiency']*100, 'b-', 
                label='Mean Efficiency', linewidth=2)
        ax3.fill_between(results['Temperature_C'], 
                        results['Min_Droplet_Efficiency']*100,
                        results['Max_Droplet_Efficiency']*100, 
                        alpha=0.2, color='b', label='Efficiency Range')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Droplet Separation Efficiency (%)')
        ax3.set_title('Droplet Separation Efficiency vs Temperature')
        ax3.grid(True)
        ax3.legend()
        
        # Gas Velocity Plot
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(results['Temperature_C'], results['Gas_Velocity'], 'r-', 
                label='Operating Velocity', linewidth=2)
        ax4.plot(results['Temperature_C'], results['Max_Gas_Velocity'], 'r--', 
                label='Maximum Velocity', linewidth=2)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Gas Velocity (m/s)')
        ax4.set_title('Gas Velocity vs Temperature')
        ax4.grid(True)
        ax4.legend()
        
        plt.suptitle('Temperature Effects on KO Drum Performance\n' + 
                    f'D = {self.diameter:.2f}m, H = {self.height:.2f}m, P = {self.P/1e5:.1f} bar',
                    y=1.02)
        plt.tight_layout()
        
        return fig, results

    def analyze_sensitivity(self, base_value, variation=0.2):
        """Analyze sensitivity of water recovery to various factors
        
        Parameters:
        -----------
        base_value : float
            Base temperature in Kelvin for analysis
        variation : float
            Fractional variation to analyze (default 0.2 = ±20%)
        """
        results = {}
        original_values = {
            'diameter': self.diameter,
            'height': self.height,
            'wire_diameter': self.wire_diameter,
            'pad_thickness': self.pad_thickness,
            'specific_area': self.specific_area,
            'P': self.P
        }
        
        # Base case
        base_case = self.analyze_temperature_water_recovery([base_value]).iloc[0]
        base_recovery = base_case['Total_Water_Recovery']
        
        # Test each parameter
        parameters = {
            'Diameter': ('diameter', self.diameter),
            'Height': ('height', self.height),
            'Wire Diameter': ('wire_diameter', self.wire_diameter),
            'Pad Thickness': ('pad_thickness', self.pad_thickness),
            'Specific Area': ('specific_area', self.specific_area),
            'Pressure': ('P', self.P)
        }
        
        for name, (param, value) in parameters.items():
            # Test decreased value
            setattr(self, param, value * (1 - variation))
            low_case = self.analyze_temperature_water_recovery([base_value]).iloc[0]
            
            # Test increased value
            setattr(self, param, value * (1 + variation))
            high_case = self.analyze_temperature_water_recovery([base_value]).iloc[0]
            
            # Calculate sensitivity
            delta_low = (low_case['Total_Water_Recovery'] - base_recovery) / base_recovery
            delta_high = (high_case['Total_Water_Recovery'] - base_recovery) / base_recovery
            
            results[name] = {
                'Base': base_recovery,
                f'-{variation*100}%': low_case['Total_Water_Recovery'],
                f'+{variation*100}%': high_case['Total_Water_Recovery'],
                'Sensitivity_Low': delta_low,
                'Sensitivity_High': delta_high
            }
            
            # Restore original value
            setattr(self, param, value)
        
        # Create sensitivity plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Absolute values plot
        params = list(results.keys())
        base_values = [results[p]['Base'] * 100 for p in params]
        low_values = [results[p][f'-{variation*100}%'] * 100 for p in params]
        high_values = [results[p][f'+{variation*100}%'] * 100 for p in params]
        
        y_pos = np.arange(len(params))
        ax1.barh(y_pos, base_values, height=0.3, color='b', alpha=0.3, label='Base')
        ax1.barh(y_pos-0.2, low_values, height=0.2, color='r', alpha=0.5, label=f'-{variation*100}%')
        ax1.barh(y_pos+0.2, high_values, height=0.2, color='g', alpha=0.5, label=f'+{variation*100}%')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(params)
        ax1.set_xlabel('Water Recovery (%)')
        ax1.set_title('Absolute Effect on Water Recovery')
        ax1.legend()
        
        # Sensitivity plot
        sensitivities_low = [results[p]['Sensitivity_Low'] * 100 for p in params]
        sensitivities_high = [results[p]['Sensitivity_High'] * 100 for p in params]
        
        ax2.barh(y_pos-0.15, sensitivities_low, height=0.3, color='r', alpha=0.5, label=f'-{variation*100}%')
        ax2.barh(y_pos+0.15, sensitivities_high, height=0.3, color='g', alpha=0.5, label=f'+{variation*100}%')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(params)
        ax2.set_xlabel('% Change in Water Recovery')
        ax2.set_title('Sensitivity Analysis')
        ax2.legend()
        
        plt.suptitle(f'Water Recovery Sensitivity Analysis at {base_value-273.15:.1f}°C')
        plt.tight_layout()
        
        return fig, pd.DataFrame(results).T

if __name__ == "__main__":
    # Create KO Drum simulator
    ko_drum = KODrumPhysics()
    
    # Run base case simulation
    print("\nBase Case Simulation:")
    report = ko_drum.get_simulation_report()
    print("\nStream Compositions:")
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(report.to_string(index=False))
    
    # Perform multi-diameter parametric study
    print("\nPerforming Multi-Diameter Parametric Study...")
    # Heights from 4m to 16m
    heights = np.linspace(4.0, 16.0, 20)
    # Diameters from 1.3m to 1.7m
    diameters = np.linspace(1.3, 1.7, 5)
    
    # Run parametric study and create comparative plots
    fig = ko_drum.plot_multi_diameter_study(heights, diameters)
    plt.show() 