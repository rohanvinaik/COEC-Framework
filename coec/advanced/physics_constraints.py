"""
Physics-Guided Constraints for COEC.

Implements deterministic physics models as constraints, following the core
COEC philosophy of separating deterministic physics from stochastic residuals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
from scipy.special import jv  # Bessel functions for diffraction

from ..core.constraint import Constraint
from ..core.substrate import Substrate


class PhysicsGuidedConstraint(Constraint):
    """
    Base class for constraints derived from physical laws.
    
    These constraints encode deterministic physics that should be
    satisfied exactly (or within measurement precision).
    """
    
    def __init__(self, 
                 name: str,
                 physics_model: Callable,
                 measurement_precision: float = 1e-6):
        super().__init__(name)
        self.physics_model = physics_model
        self.measurement_precision = measurement_precision
        
        # Cache for physics computations
        self._physics_cache = {}
    
    def satisfaction(self, state: np.ndarray) -> float:
        """
        Compute how well state satisfies physics constraint.
        
        Returns 1.0 if physics is satisfied within measurement precision,
        lower values indicate violation.
        """
        # Compute physics prediction
        physics_prediction = self.physics_model(state)
        
        # Compute deviation from physics
        if isinstance(physics_prediction, dict):
            # Multiple physics quantities
            total_deviation = 0.0
            for key, predicted in physics_prediction.items():
                actual = self._extract_quantity(state, key)
                deviation = np.mean(np.abs(actual - predicted))
                total_deviation += deviation
            
            avg_deviation = total_deviation / len(physics_prediction)
        else:
            # Single physics quantity
            avg_deviation = np.mean(np.abs(state - physics_prediction))
        
        # Map deviation to satisfaction score
        if avg_deviation <= self.measurement_precision:
            return 1.0
        else:
            # Exponential decay based on deviation
            return np.exp(-avg_deviation / self.measurement_precision)
    
    def _extract_quantity(self, state: np.ndarray, quantity_name: str) -> np.ndarray:
        """Extract named physical quantity from state."""
        # Override in subclasses for specific extraction logic
        return state
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gradient to guide system toward physics satisfaction.
        
        Uses finite differences by default, but can be overridden for
        analytical gradients.
        """
        epsilon = 1e-6
        grad = np.zeros_like(state)
        base_satisfaction = self.satisfaction(state)
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                state[i, j] += epsilon
                perturbed_satisfaction = self.satisfaction(state)
                grad[i, j] = (perturbed_satisfaction - base_satisfaction) / epsilon
                state[i, j] -= epsilon
        
        return grad


class LensPhysicsModel:
    """
    Complete physics model for optical lens systems.
    
    Implements deterministic models for distortion, chromatic aberration,
    vignetting, and diffraction as used in VintageOptics.
    """
    
    def __init__(self, lens_parameters: Optional[Dict[str, Any]] = None):
        # Default parameters for a typical vintage lens
        self.params = lens_parameters or {
            # Brown-Conrady distortion coefficients
            'k1': -0.15,  # Barrel distortion
            'k2': 0.05,   # Higher order
            'k3': -0.01,
            'p1': 0.001,  # Tangential
            'p2': 0.0005,
            
            # Chromatic aberration
            'lateral_chromatic': 0.002,
            'longitudinal_chromatic': 0.02,
            
            # Vignetting
            'vignetting_strength': 0.3,
            'vignetting_radius': 0.8,
            
            # Diffraction
            'aperture': 2.8,
            'focal_length': 50,  # mm
            'wavelength': 550e-9,  # Green light (m)
            
            # Bokeh shape
            'aperture_blades': 6,
            'blade_curvature': 0.8
        }
    
    def apply_brown_conrady_distortion(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply Brown-Conrady lens distortion model.
        
        Args:
            coords: (N, 2) array of normalized coordinates in [-1, 1]
            
        Returns:
            Distorted coordinates
        """
        x, y = coords[:, 0], coords[:, 1]
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3
        
        # Radial distortion
        radial = 1 + self.params['k1']*r2 + self.params['k2']*r4 + self.params['k3']*r6
        
        # Tangential distortion
        dx = 2*self.params['p1']*x*y + self.params['p2']*(r2 + 2*x**2)
        dy = self.params['p1']*(r2 + 2*y**2) + 2*self.params['p2']*x*y
        
        # Apply distortion
        x_distorted = x*radial + dx
        y_distorted = y*radial + dy
        
        return np.column_stack([x_distorted, y_distorted])
    
    def compute_chromatic_shift(self, coords: np.ndarray, wavelength: float) -> np.ndarray:
        """
        Compute chromatic aberration shift for given wavelength.
        
        Uses simplified model where shift is proportional to distance from center
        and wavelength deviation from reference (550nm green).
        """
        x, y = coords[:, 0], coords[:, 1]
        r = np.sqrt(x**2 + y**2)
        
        # Wavelength deviation from green
        reference_wavelength = 550e-9
        wavelength_factor = (wavelength - reference_wavelength) / reference_wavelength
        
        # Lateral chromatic aberration
        lateral_shift = self.params['lateral_chromatic'] * wavelength_factor * r
        
        # Apply shift radially
        angle = np.arctan2(y, x)
        dx = lateral_shift * np.cos(angle)
        dy = lateral_shift * np.sin(angle)
        
        return np.column_stack([x + dx, y + dy])
    
    def compute_vignetting(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute vignetting (darkening toward edges) using cos^4 law.
        
        Returns multiplier in [0, 1] for each coordinate.
        """
        x, y = coords[:, 0], coords[:, 1]
        r = np.sqrt(x**2 + y**2)
        
        # Normalize radius by vignetting radius
        r_normalized = r / self.params['vignetting_radius']
        
        # Cos^4 falloff
        cos_angle = np.maximum(0, 1 - r_normalized)
        vignetting = 1 - self.params['vignetting_strength'] * (1 - cos_angle**4)
        
        return np.clip(vignetting, 0, 1)
    
    def compute_airy_disk_psf(self, size: int = 33) -> np.ndarray:
        """
        Compute Point Spread Function for diffraction (Airy disk pattern).
        
        Returns 2D PSF kernel for convolution.
        """
        # Calculate Airy disk radius
        wavelength = self.params['wavelength']
        f_number = self.params['aperture']
        airy_radius = 1.22 * wavelength * f_number / 1e-6  # Convert to pixels
        
        # Create coordinate grid
        center = size // 2
        y, x = np.ogrid[-center:size-center, -center:size-center]
        r = np.sqrt(x**2 + y**2)
        
        # Compute Airy pattern
        # J1 is first-order Bessel function
        with np.errstate(divide='ignore', invalid='ignore'):
            pattern = np.where(r == 0, 1, 2 * jv(1, np.pi * r / airy_radius) / (np.pi * r / airy_radius))
        
        # Square for intensity
        psf = pattern**2
        
        # Normalize
        psf = psf / np.sum(psf)
        
        return psf
    
    def compute_bokeh_kernel(self, size: int = 33) -> np.ndarray:
        """
        Compute bokeh kernel based on aperture shape.
        
        Returns 2D kernel representing out-of-focus point spread.
        """
        n_blades = self.params['aperture_blades']
        curvature = self.params['blade_curvature']
        
        # Create coordinate grid
        center = size // 2
        y, x = np.ogrid[-center:size-center, -center:size-center]
        
        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2) / center
        theta = np.arctan2(y, x)
        
        # Create polygon aperture shape
        # Distance from center to edge at angle theta
        blade_angle = 2 * np.pi / n_blades
        
        # Find nearest blade edge
        blade_phase = theta % blade_angle
        edge_distance = np.cos(blade_phase - blade_angle/2) / np.cos(blade_angle/2)
        
        # Apply blade curvature
        edge_distance = edge_distance * (1 - curvature * (1 - edge_distance))
        
        # Create aperture mask
        kernel = np.where(r <= edge_distance, 1.0, 0.0)
        
        # Smooth edges slightly
        from scipy.ndimage import gaussian_filter
        kernel = gaussian_filter(kernel, sigma=0.5)
        
        # Normalize
        kernel = kernel / np.sum(kernel)
        
        return kernel


class ConservationLawConstraint(PhysicsGuidedConstraint):
    """
    Constraint enforcing physical conservation laws.
    
    Examples: energy conservation, mass conservation, momentum conservation.
    """
    
    def __init__(self,
                 name: str,
                 conserved_quantity: str,
                 extraction_function: Callable[[np.ndarray], float],
                 target_value: float):
        
        def conservation_model(state):
            current_value = extraction_function(state)
            return target_value  # Physics says it should equal target
        
        super().__init__(name, conservation_model)
        self.conserved_quantity = conserved_quantity
        self.extraction_function = extraction_function
        self.target_value = target_value
    
    def satisfaction(self, state: np.ndarray) -> float:
        """Check if conservation law is satisfied."""
        current_value = self.extraction_function(state)
        deviation = abs(current_value - self.target_value)
        relative_deviation = deviation / (abs(self.target_value) + 1e-10)
        
        # Very strict for conservation laws
        if relative_deviation < 1e-6:
            return 1.0
        else:
            return np.exp(-100 * relative_deviation)  # Sharp penalty


class ThermodynamicConstraint(PhysicsGuidedConstraint):
    """
    Constraints based on thermodynamic laws.
    
    Ensures states respect entropy increase, free energy minimization, etc.
    """
    
    def __init__(self, name: str, temperature: float = 300.0):
        self.temperature = temperature
        self.boltzmann_constant = 1.380649e-23
        
        def thermodynamic_model(state):
            # Compute free energy components
            energy = self._compute_energy(state)
            entropy = self._compute_entropy(state)
            
            # Free energy F = E - TS
            free_energy = energy - temperature * entropy
            
            return {
                'free_energy': free_energy,
                'entropy': entropy,
                'energy': energy
            }
        
        super().__init__(name, thermodynamic_model)
    
    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute system energy (placeholder - override in subclasses)."""
        # Example: quadratic energy
        return 0.5 * np.sum(state**2)
    
    def _compute_entropy(self, state: np.ndarray) -> float:
        """Compute system entropy."""
        # Discretize state for probability distribution
        hist, _ = np.histogram(state.flatten(), bins=50)
        probs = hist / np.sum(hist)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy


class FluidDynamicsConstraint(PhysicsGuidedConstraint):
    """
    Constraints based on fluid dynamics (Navier-Stokes).
    
    Useful for problems involving flow, diffusion, or transport.
    """
    
    def __init__(self,
                 name: str,
                 viscosity: float = 1e-3,
                 density: float = 1000.0):
        self.viscosity = viscosity
        self.density = density
        
        def navier_stokes_model(state):
            # State assumed to be velocity field
            # Compute required derivatives
            dudx = np.gradient(state[:, :, 0], axis=0) if state.ndim > 2 else np.gradient(state[:, 0])
            dudy = np.gradient(state[:, :, 0], axis=1) if state.ndim > 2 else 0
            
            # Simplified incompressible flow
            divergence = dudx + dudy
            
            # Should be zero for incompressible flow
            return {'divergence': np.zeros_like(divergence)}
        
        super().__init__(name, navier_stokes_model)


class SymmetryConstraint(PhysicsGuidedConstraint):
    """
    Constraint enforcing physical symmetries.
    
    Examples: rotational symmetry, translational invariance, time reversal.
    """
    
    def __init__(self,
                 name: str,
                 symmetry_type: str,
                 symmetry_operation: Callable[[np.ndarray], np.ndarray]):
        self.symmetry_type = symmetry_type
        self.symmetry_operation = symmetry_operation
        
        def symmetry_model(state):
            # Apply symmetry operation
            transformed = symmetry_operation(state)
            # Physics says original and transformed should be identical
            return transformed
        
        super().__init__(name, symmetry_model)
    
    def satisfaction(self, state: np.ndarray) -> float:
        """Check if state is symmetric under operation."""
        transformed = self.symmetry_operation(state)
        difference = np.mean(np.abs(state - transformed))
        
        if difference < 1e-6:
            return 1.0
        else:
            return np.exp(-10 * difference)


def create_physics_constraint_library() -> Dict[str, PhysicsGuidedConstraint]:
    """
    Create library of common physics constraints.
    
    Returns dictionary of ready-to-use physics constraints.
    """
    constraints = {}
    
    # Conservation of energy
    def extract_kinetic_energy(state):
        # Assume state represents velocities
        return 0.5 * np.sum(state**2)
    
    constraints['energy_conservation'] = ConservationLawConstraint(
        'energy_conservation',
        'kinetic_energy',
        extract_kinetic_energy,
        target_value=1.0  # Normalized
    )
    
    # Rotational symmetry
    def rotate_90(state):
        if state.ndim == 2:
            return np.rot90(state)
        return state
    
    constraints['rotational_symmetry'] = SymmetryConstraint(
        'rotational_symmetry',
        'rotation_90',
        rotate_90
    )
    
    # Thermodynamic equilibrium
    constraints['thermal_equilibrium'] = ThermodynamicConstraint(
        'thermal_equilibrium',
        temperature=300.0
    )
    
    # Incompressible flow
    constraints['incompressible_flow'] = FluidDynamicsConstraint(
        'incompressible_flow',
        viscosity=1e-3
    )
    
    return constraints
