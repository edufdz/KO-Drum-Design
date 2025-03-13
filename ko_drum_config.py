import numpy as np

# KO Drum Configuration File

# Operating Conditions
OPERATING_CONDITIONS = {
    'pressure': 1.01325,  # bar (1.01325 bar = 1 atm)
    'pressure_Pa': 101325,  # Pa
    'temperature': 50,  # °C
    'temperature_K': 323.15,  # K (50°C + 273.15)
}

# Vessel Dimensions (based on typical L/D ratio of 3-4)
VESSEL_DIMENSIONS = {
    'diameter': 1.45,  # meters
    'height': 5,  # meters
    'liquid_holdup_height': 0.51,  # meters
    'total_volume': np.pi * 1.45**2 * 5 / 4,  # m³ (π * d² * h / 4)
}

# Component Properties
MOLECULAR_WEIGHTS = {
    'CH4': 16.04,    # kg/kmol
    'H2O': 18.015,   # kg/kmol
    'CO2': 44.01,    # kg/kmol
    'CO': 28.01,     # kg/kmol
    'H2': 2.016      # kg/kmol
}

# Feed Composition (from image)
FEED_COMPOSITION = {
    'CH4': 15.36,    # kmol/h
    'H2O': 92.48,    # kmol/h
    'CO2': 84.59,    # kmol/h
    'CO': 84.59,     # kmol/h
    'H2': 443.12     # kmol/h
}

# Total molar flow
TOTAL_MOLAR_FLOW = 716.66  # kmol/h (sum of all components)

# Phase Distribution (vapor fraction)
VAPOR_FRACTION = {
    'CH4': 1.0,    # Complete vapor
    'H2O': 0.05,   # 5% in vapor phase, 95% condensed
    'CO2': 1.0,    # Complete vapor
    'CO': 1.0,     # Complete vapor
    'H2': 1.0      # Complete vapor
}

# Energy Flow
ENERGY_PARAMETERS = {
    'total_enthalpy_flow': -17.17,  # MW (from image)
    'reference_temperature': 25,  # °C (typical reference temperature)
    'heat_loss_factor': 0.02  # 2% heat loss assumption
}

# Design Parameters
DESIGN_PARAMETERS = {
    'residence_time': 5,  # minutes (typical for KO drums)
    'design_pressure': 1.5,  # bar (typical safety factor of 1.5)
    'design_temperature': 80,  # °C (typical design margin)
    'min_liquid_level': 0.2,  # m (minimum liquid level for pump protection)
    'max_liquid_level': 0.8,  # m (maximum normal operating level)
    'gas_velocity_limit': 4.0  # m/s (typical maximum gas velocity)
}

# Separation Efficiency
SEPARATION_EFFICIENCY = {
    'droplet_size_cutoff': 10,  # microns
    'efficiency_above_cutoff': 0.99,  # 99% removal above cutoff size
    'overall_efficiency': 0.95  # 95% overall separation efficiency
}

# Material Properties
MATERIAL_PROPERTIES = {
    'vessel_material': 'Carbon Steel',
    'design_stress': 137.9,  # MPa (typical for carbon steel)
    'corrosion_allowance': 3.0,  # mm
    'joint_efficiency': 0.85  # typical for welded vessels
}

# Control Parameters
CONTROL_PARAMETERS = {
    'level_control_type': 'PID',
    'level_control_setpoint': 0.5,  # m
    'pressure_control_setpoint': 1.01325,  # bar
    'temperature_control_range': (45, 55)  # °C
}

def calculate_mass_flows():
    """Calculate mass flows for each component"""
    mass_flows = {}
    for component, molar_flow in FEED_COMPOSITION.items():
        mass_flows[component] = molar_flow * MOLECULAR_WEIGHTS[component]
    return mass_flows

def get_total_mass_flow():
    """Calculate total mass flow"""
    mass_flows = calculate_mass_flows()
    return sum(mass_flows.values())

# Calculate and store mass flows
MASS_FLOWS = calculate_mass_flows()
TOTAL_MASS_FLOW = get_total_mass_flow()

if __name__ == "__main__":
    print("KO Drum Configuration Summary:")
    print(f"Total Mass Flow: {TOTAL_MASS_FLOW:.2f} kg/h")
    print(f"Operating Temperature: {OPERATING_CONDITIONS['temperature']}°C")
    print(f"Operating Pressure: {OPERATING_CONDITIONS['pressure']} bar")
    print("\nComponent Mass Flows (kg/h):")
    for component, flow in MASS_FLOWS.items():
        print(f"{component}: {flow:.2f}") 