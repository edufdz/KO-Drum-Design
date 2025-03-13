import numpy as np
from CoolProp.CoolProp import PropsSI
import pandas as pd

class KODrumPhysicalProperties:
    def __init__(self, T, P, feed, M_W):
        self.T = T  # Temperature in K
        self.P = P  # Pressure in Pa
        self.feed = feed
        self.M_W = M_W
        
        # Critical properties for water
        self.Tc = 647.15  # K
        self.Pc = 22.064e6  # Pa
        self.omega = 0.344  # Acentric factor
        self.R = 8.314  # Gas constant
        
        # Antoine equation parameters for water (valid from 1-100°C)
        self.A_antoine = 8.07131
        self.B_antoine = 1730.63
        self.C_antoine = 233.426
        
        # Initialize properties
        self.gas_density = None
        self.liquid_density = None
        self.gas_viscosity = None
        self.liquid_viscosity = None
        self.surface_tension = None
        
    def calculate_compressibility(self):
        """Calculate gas compressibility using Peng-Robinson EOS"""
        Tr = self.T / self.Tc
        Pr = self.P / self.Pc
        
        # Calculate a and b parameters
        kappa = 0.37464 + 1.54226*self.omega - 0.26992*self.omega**2
        alpha = (1 + kappa*(1 - np.sqrt(Tr)))**2
        
        a = 0.45724 * (self.R*self.Tc)**2 / self.Pc * alpha
        b = 0.07780 * self.R*self.Tc / self.Pc
        
        # Calculate A and B
        A = a*self.P / (self.R*self.T)**2
        B = b*self.P / (self.R*self.T)
        
        # Solve cubic equation for Z
        return self._solve_cubic_PR(A, B)
    
    def _solve_cubic_PR(self, A, B):
        """Solve Peng-Robinson cubic equation"""
        coeff = [1, -(1-B), A-3*B**2-2*B, -(A*B-B**2-B**3)]
        roots = np.roots(coeff)
        # Select the maximum real root for vapor phase
        Z = np.real(roots[np.argmax(np.real(roots))])
        return Z
    
    def calculate_vapor_pressure(self):
        """Calculate water vapor pressure using Antoine equation"""
        T_C = self.T - 273.15
        P_vap_mmHg = 10**(self.A_antoine - self.B_antoine/(T_C + self.C_antoine))
        return P_vap_mmHg * 133.322  # Convert to Pa

    def calculate_water_composition(self):
        """Calculate vapor-liquid composition for water"""
        P_vap = self.calculate_vapor_pressure()
        y_water = min(P_vap / self.P, 1.0)  # Vapor fraction using Dalton's law
        return {
            'vapor_fraction': y_water,
            'liquid_fraction': 1.0,  # Pure water in liquid phase
            'vapor_pressure': P_vap
        }

    def calculate_fluid_properties(self):
        """Calculate temperature and pressure dependent fluid properties"""
        # Get water composition for density calculations
        water_comp = self.calculate_water_composition()
        
        # Calculate gas density using real gas law
        Z = self.calculate_compressibility()
        self.gas_density = self.P * self.M_W['H2O'] / (Z * self.R * self.T)
        
        # Calculate liquid density using modified Rackett equation
        Tr = self.T / self.Tc
        self.liquid_density = 1000 * (1 + 0.1*(1-Tr)**(0.33))
        
        # Calculate viscosities
        self.gas_viscosity = self._calculate_gas_viscosity()
        self.liquid_viscosity = self._calculate_liquid_viscosity()
        
        # Calculate surface tension
        self.surface_tension = self._calculate_surface_tension(Tr)
        
        return {
            'gas_density': self.gas_density,
            'liquid_density': self.liquid_density,
            'gas_viscosity': self.gas_viscosity,
            'liquid_viscosity': self.liquid_viscosity,
            'surface_tension': self.surface_tension,
            'Z_factor': Z,
            'vapor_fraction': water_comp['vapor_fraction']
        }
    
    def _calculate_gas_viscosity(self):
        """Calculate gas viscosity using Sutherland's formula"""
        mu_0 = 1.8e-5
        T_0 = 293.15
        S = 110.4
        return mu_0 * (self.T/T_0)**(3/2) * (T_0 + S)/(self.T + S)
    
    def _calculate_liquid_viscosity(self):
        """Calculate liquid viscosity using modified Andrade equation"""
        return 2.414e-5 * 10**(247.8/(self.T-140))
    
    def _calculate_surface_tension(self, Tr):
        """Calculate surface tension using correlation"""
        return 0.2358 * (1 - Tr)**(1.256) * (1 - 0.625*(1-Tr))
    
    def calculate_thermal_properties(self):
        """Calculate thermal properties"""
        Cp_gas = 1996  # J/kg-K for water vapor
        self.thermal_conductivity = (1.32 + 1.77/self.M_W['H2O']) * self.gas_viscosity * Cp_gas
        self.thermal_diffusivity = self.thermal_conductivity / (self.gas_density * Cp_gas)
        
        return {
            'thermal_conductivity': self.thermal_conductivity,
            'thermal_diffusivity': self.thermal_diffusivity,
            'Cp': Cp_gas
        }
    
    def calculate_droplet_physics(self, droplet_diameter, gas_velocity):
        """Calculate droplet physics parameters"""
        Re = self.gas_density * gas_velocity * droplet_diameter / self.gas_viscosity
        We = self.gas_density * gas_velocity**2 * droplet_diameter / self.surface_tension
        Fr = gas_velocity**2 / (9.81 * droplet_diameter)
        
        Cd = self._calculate_drag_coefficient(Re)
        settling_velocity = np.sqrt((4 * 9.81 * droplet_diameter * 
                                   (self.liquid_density - self.gas_density)) / 
                                  (3 * Cd * self.gas_density))
        
        return {
            'reynolds': Re,
            'weber': We,
            'froude': Fr,
            'drag_coefficient': Cd,
            'settling_velocity': settling_velocity
        }
    
    def _calculate_drag_coefficient(self, Re):
        """Calculate drag coefficient based on Reynolds number"""
        if Re < 0.1:
            return 24/Re  # Stokes regime
        elif Re < 1000:
            return 24/Re * (1 + 0.15*Re**0.687)  # Intermediate regime
        return 0.44  # Newton regime

    def calculate_detailed_water_composition(self, total_water_mass=1.0, system_volume=1.0):
        """Calculate detailed water phase composition"""
        P_vap = self.calculate_vapor_pressure()
        y_water = min(P_vap / self.P, 1.0)
        
        M_H2O = 18.015  # g/mol
        n_total = total_water_mass / M_H2O
        n_vapor = n_total * y_water
        n_liquid = n_total - n_vapor
        
        m_vapor = n_vapor * M_H2O
        m_liquid = n_liquid * M_H2O
        
        return {
            'temperature_C': self.T - 273.15,
            'vapor_pressure_Pa': P_vap,
            'vapor_pressure_bar': P_vap / 1e5,
            'moles_vapor': n_vapor,
            'moles_liquid': n_liquid,
            'mass_vapor_kg': m_vapor,
            'mass_liquid_kg': m_liquid,
            'vapor_percent': (m_vapor / total_water_mass) * 100,
            'liquid_percent': (m_liquid / total_water_mass) * 100,
            'vapor_density_kg_m3': m_vapor / system_volume,
            'liquid_density_kg_m3': m_liquid / system_volume,
            'total_mass_kg': total_water_mass,
            'vapor_mole_fraction': y_water,
            'liquid_mole_fraction': 1.0
        }

    def analyze_water_phase_diagram(self, temperatures, total_water_mass=1.0, system_volume=1.0):
        """Analyze water phase composition across a range of temperatures"""
        results = []
        original_T = self.T
        
        for T in temperatures:
            self.T = T
            results.append(self.calculate_detailed_water_composition(total_water_mass, system_volume))
        
        self.T = original_T
        return pd.DataFrame(results)

    def calculate_enthalpy(self, component):
        """Calculate specific enthalpy for a component"""
        if component == 'H2O':
            # Water vapor enthalpy using simplified correlation
            Cp_gas = 1996  # J/kg-K (average value)
            h_vap = 2257e3  # J/kg at normal boiling point
            return Cp_gas * (self.T - 273.15) + h_vap
        else:
            # For other components use ideal gas approximation
            Cp = 1000  # J/kg-K (approximate)
            return Cp * (self.T - 273.15)

    def calculate_dimensionless_numbers(self, gas_velocity, droplet_diameter):
        """Calculate dimensionless numbers for the system"""
        droplet_physics = self.calculate_droplet_physics(droplet_diameter, gas_velocity)
        return {
            'reynolds': droplet_physics['reynolds'],
            'weber': droplet_physics['weber'],
            'froude': droplet_physics['froude']
        }

    def calculate_phase_properties(self):
        """Calculate properties for both phases including composition effects"""
        # Get base properties
        props = self.calculate_fluid_properties()
        
        # Get water composition
        water_comp = self.calculate_water_composition()
        
        # Calculate actual densities with composition
        vapor_density = props['gas_density'] * water_comp['vapor_fraction']
        liquid_density = props['liquid_density'] * water_comp['liquid_fraction']
        
        return {
            'vapor_phase': {
                'density': vapor_density,
                'composition': water_comp['vapor_fraction'],
                'pressure': water_comp['vapor_pressure']
            },
            'liquid_phase': {
                'density': liquid_density,
                'composition': water_comp['liquid_fraction']
            },
            'temperature_C': self.T - 273.15
        } 