import numpy as np
from scipy.optimize import fsolve
from CoolProp.CoolProp import PropsSI

class PengRobinson:
    def __init__(self):
        """Initialize Peng-Robinson EOS calculator with component properties"""
        # Critical properties and acentric factors for common components
        self.components = {
            'H2O': {'Tc': 647.14, 'Pc': 22.064e6, 'omega': 0.344},
            'CO2': {'Tc': 304.13, 'Pc': 7.377e6, 'omega': 0.239},
            'CH4': {'Tc': 190.56, 'Pc': 4.599e6, 'omega': 0.011},
            'H2': {'Tc': 33.19, 'Pc': 1.315e6, 'omega': -0.216},
            'CO': {'Tc': 132.85, 'Pc': 3.494e6, 'omega': 0.048}
        }
        
        # Binary interaction parameters (kij)
        self.kij = {
            ('H2O', 'CO2'): 0.19,
            ('H2O', 'CH4'): 0.49,
            ('H2O', 'H2'): 0.28,
            ('H2O', 'CO'): 0.07,
            ('CO2', 'CH4'): 0.10,
            ('CO2', 'H2'): 0.10,
            ('CO2', 'CO'): 0.05,
            ('CH4', 'H2'): 0.08,
            ('CH4', 'CO'): 0.02,
            ('H2', 'CO'): 0.03
        }
        
        # Gas constant
        self.R = 8.314  # J/mol·K
    
    def get_kij(self, comp_i, comp_j):
        """Get binary interaction parameter"""
        if comp_i == comp_j:
            return 0.0
        key = tuple(sorted([comp_i, comp_j]))
        return self.kij.get(key, 0.0)
    
    def calculate_a_b(self, T, comp):
        """Calculate 'a' and 'b' parameters for a component"""
        Tc = self.components[comp]['Tc']
        Pc = self.components[comp]['Pc']
        omega = self.components[comp]['omega']
        
        # Calculate alpha parameter
        Tr = T / Tc
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
        alpha = (1 + kappa*(1 - np.sqrt(Tr)))**2
        
        # Calculate a and b
        a = 0.45724 * (self.R * Tc)**2 / Pc * alpha
        b = 0.07780 * self.R * Tc / Pc
        
        return a, b
    
    def calculate_mixture_parameters(self, T, composition):
        """Calculate mixture parameters for PR-EOS"""
        a_mix = 0.0
        b_mix = 0.0
        
        # Calculate individual a and b parameters
        params = {}
        for comp in composition:
            params[comp] = self.calculate_a_b(T, comp)
        
        # Calculate mixture parameters using mixing rules
        for comp_i in composition:
            x_i = composition[comp_i]
            b_mix += x_i * params[comp_i][1]
            
            for comp_j in composition:
                x_j = composition[comp_j]
                a_ij = np.sqrt(params[comp_i][0] * params[comp_j][0]) * \
                       (1 - self.get_kij(comp_i, comp_j))
                a_mix += x_i * x_j * a_ij
        
        return a_mix, b_mix
    
    def calculate_Z_factor(self, T, P, composition):
        """Calculate compressibility factor using PR-EOS"""
        a_mix, b_mix = self.calculate_mixture_parameters(T, composition)
        
        # Convert coefficients for cubic equation
        A = a_mix * P / (self.R * T)**2
        B = b_mix * P / (self.R * T)
        
        # Cubic equation coefficients: Z³ + pZ² + qZ + r = 0
        p = -1 + B
        q = A - 3*B**2 - 2*B
        r = -A*B + B**2 + B**3
        
        # Solve cubic equation
        coeffs = [1, p, q, r]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        
        # Sort roots: largest is vapor, smallest is liquid
        Z_vapor = np.max(real_roots)
        Z_liquid = np.min(real_roots)
        
        return Z_vapor, Z_liquid
    
    def calculate_fugacity_coefficient(self, T, P, composition, phase='vapor'):
        """Calculate fugacity coefficients for each component"""
        a_mix, b_mix = self.calculate_mixture_parameters(T, composition)
        Z = self.calculate_Z_factor(T, P, composition)[1 if phase == 'liquid' else 0]
        
        phi = {}
        for comp_i in composition:
            # Calculate partial derivatives
            sum_term = 0
            for comp_j in composition:
                a_ij = np.sqrt(self.calculate_a_b(T, comp_i)[0] * 
                             self.calculate_a_b(T, comp_j)[0]) * \
                       (1 - self.get_kij(comp_i, comp_j))
                sum_term += composition[comp_j] * a_ij
            
            b_i = self.calculate_a_b(T, comp_i)[1]
            
            # Calculate fugacity coefficient
            term1 = b_i/b_mix * (Z - 1)
            term2 = np.log(Z - b_mix*P/(self.R*T))
            term3 = a_mix/(2*np.sqrt(2)*b_mix*self.R*T) * \
                   (2*sum_term/a_mix - b_i/b_mix) * \
                   np.log((Z + (1+np.sqrt(2))*b_mix*P/(self.R*T)) / 
                         (Z + (1-np.sqrt(2))*b_mix*P/(self.R*T)))
            
            phi[comp_i] = np.exp(term1 - term2 - term3)
        
        return phi
    
    def calculate_K_values(self, T, P, composition):
        """Calculate K-values using fugacity coefficients"""
        phi_V = self.calculate_fugacity_coefficient(T, P, composition, 'vapor')
        phi_L = self.calculate_fugacity_coefficient(T, P, composition, 'liquid')
        
        K_values = {}
        for comp in composition:
            K_values[comp] = phi_V[comp] / phi_L[comp]
        
        return K_values
    
    def calculate_density(self, T, P, composition, phase='vapor'):
        """Calculate density using PR-EOS"""
        Z = self.calculate_Z_factor(T, P, composition)[1 if phase == 'liquid' else 0]
        
        # Calculate average molecular weight
        M_avg = sum(composition[comp] * PropsSI('M', comp) for comp in composition)
        
        # Calculate density (kg/m³)
        density = P * M_avg / (Z * self.R * T)
        return density
    
    def calculate_enthalpy_departure(self, T, P, composition, phase='vapor'):
        """Calculate enthalpy departure using PR-EOS"""
        a_mix, b_mix = self.calculate_mixture_parameters(T, composition)
        Z = self.calculate_Z_factor(T, P, composition)[1 if phase == 'liquid' else 0]
        
        # Calculate temperature derivative of a_mix
        da_dT = 0
        for comp_i in composition:
            for comp_j in composition:
                x_i = composition[comp_i]
                x_j = composition[comp_j]
                
                # Calculate temperature derivatives of individual a parameters
                Tc_i = self.components[comp_i]['Tc']
                Pc_i = self.components[comp_i]['Pc']
                omega_i = self.components[comp_i]['omega']
                kappa_i = 0.37464 + 1.54226*omega_i - 0.26992*omega_i**2
                
                Tc_j = self.components[comp_j]['Tc']
                Pc_j = self.components[comp_j]['Pc']
                omega_j = self.components[comp_j]['omega']
                kappa_j = 0.37464 + 1.54226*omega_j - 0.26992*omega_j**2
                
                da_i_dT = -0.45724 * (self.R * Tc_i)**2 / Pc_i * kappa_i / np.sqrt(T/Tc_i)
                da_j_dT = -0.45724 * (self.R * Tc_j)**2 / Pc_j * kappa_j / np.sqrt(T/Tc_j)
                
                da_dT += x_i * x_j * np.sqrt(da_i_dT * da_j_dT) * \
                         (1 - self.get_kij(comp_i, comp_j))
        
        # Calculate enthalpy departure
        term1 = self.R * T * (Z - 1)
        term2 = (T*da_dT - a_mix)/(2*np.sqrt(2)*b_mix) * \
                np.log((Z + (1+np.sqrt(2))*b_mix*P/(self.R*T)) / 
                      (Z + (1-np.sqrt(2))*b_mix*P/(self.R*T)))
        
        H_departure = term1 + term2
        return H_departure

# Example usage
if __name__ == "__main__":
    # Test the Peng-Robinson EOS
    pr = PengRobinson()
    
    # Example conditions
    T = 323.15  # K
    P = 101325  # Pa
    compositions = {
        'CH4': 0.02,
        'H2O': 0.13,
        'CO2': 0.12,
        'CO': 0.12,
        'H2': 0.61
    }
    
    # Calculate K-values
    K_values = pr.calculate_K_values(T, P, compositions)
    print("\nK-values:")
    for comp, K in K_values.items():
        print(f"{comp}: {K:.3f}")
    
    # Calculate Z-factors
    Z_vapor, Z_liquid = pr.calculate_Z_factor(T, P, compositions)
    print(f"\nCompressibility factors:")
    print(f"Z_vapor: {Z_vapor:.3f}")
    print(f"Z_liquid: {Z_liquid:.3f}")
    
    # Calculate densities
    rho_vapor = pr.calculate_density(T, P, compositions, 'vapor')
    rho_liquid = pr.calculate_density(T, P, compositions, 'liquid')
    print(f"\nDensities:")
    print(f"Vapor: {rho_vapor:.2f} kg/m³")
    print(f"Liquid: {rho_liquid:.2f} kg/m³") 