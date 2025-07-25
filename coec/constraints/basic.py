"""
Implementation of various constraint types for COEC systems.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from ..core.system import Constraint


class EnergeticConstraint(Constraint):
    """
    Energetic constraints that bias the energy landscape.
    
    These constraints favor states with lower potential energy according
    to a specified potential function.
    """
    
    def __init__(self, 
                 name: str = "energetic",
                 precision: float = 1.0,
                 potential: str = "lennard_jones",
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(name, precision)
        self.potential = potential
        self.parameters = parameters or {}
        
        # Set up potential function
        if potential == "lennard_jones":
            self.potential_func = self._lennard_jones_potential
        elif potential == "harmonic":
            self.potential_func = self._harmonic_potential
        else:
            raise ValueError(f"Unknown potential: {potential}")
    
    def _lennard_jones_potential(self, distances: np.ndarray) -> np.ndarray:
        """Lennard-Jones 6-12 potential."""
        epsilon = self.parameters.get("epsilon", 1.0)
        sigma = self.parameters.get("sigma", 1.0)
        
        r6 = (sigma / distances) ** 6
        return 4 * epsilon * (r6 ** 2 - r6)
    
    def _harmonic_potential(self, distances: np.ndarray) -> np.ndarray:
        """Harmonic potential."""
        k = self.parameters.get("k", 1.0)
        r0 = self.parameters.get("r0", 1.0)
        
        return 0.5 * k * (distances - r0) ** 2
    
    def satisfaction(self, state: np.ndarray) -> float:
        """
        Compute satisfaction based on energy.
        
        Lower energy states have higher satisfaction.
        """
        # Compute pairwise distances
        n_particles = state.shape[0]
        total_energy = 0.0
        
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                distance = np.linalg.norm(state[i] - state[j])
                if distance > 0:
                    total_energy += self.potential_func(distance)
        
        # Convert energy to satisfaction score [0, 1]
        # Using sigmoid-like transformation
        max_energy = self.parameters.get("max_energy", 100.0)
        satisfaction = 1.0 / (1.0 + np.exp(total_energy / max_energy))
        
        return satisfaction
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute gradient of energetic constraint."""
        # Numerical gradient for now
        epsilon = 1e-6
        grad = np.zeros_like(state)
        base_satisfaction = self.satisfaction(state)
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                state[i, j] += epsilon
                grad[i, j] = (self.satisfaction(state) - base_satisfaction) / epsilon
                state[i, j] -= epsilon
        
        return grad


class TopologicalConstraint(Constraint):
    """
    Topological constraints that restrict connectivity or spatial arrangement.
    """
    
    def __init__(self,
                 name: str = "topological",
                 precision: float = 1.0,
                 connectivity: str = "chain",
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(name, precision)
        self.connectivity = connectivity
        self.parameters = parameters or {}
        
        if connectivity == "chain":
            self.check_connectivity = self._check_chain_connectivity
        elif connectivity == "fully_connected":
            self.check_connectivity = self._check_full_connectivity
        else:
            raise ValueError(f"Unknown connectivity: {connectivity}")
    
    def _check_chain_connectivity(self, state: np.ndarray) -> float:
        """Check if particles form a chain."""
        n_particles = state.shape[0]
        bond_length = self.parameters.get("bond_length", 1.0)
        tolerance = self.parameters.get("tolerance", 0.2)
        
        satisfaction = 0.0
        for i in range(n_particles - 1):
            distance = np.linalg.norm(state[i] - state[i + 1])
            if abs(distance - bond_length) < tolerance:
                satisfaction += 1.0
        
        return satisfaction / (n_particles - 1)
    
    def _check_full_connectivity(self, state: np.ndarray) -> float:
        """Check if all particles are within interaction range."""
        n_particles = state.shape[0]
        max_distance = self.parameters.get("max_distance", 5.0)
        
        connected_pairs = 0
        total_pairs = n_particles * (n_particles - 1) / 2
        
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                distance = np.linalg.norm(state[i] - state[j])
                if distance <= max_distance:
                    connected_pairs += 1
        
        return connected_pairs / total_pairs
    
    def satisfaction(self, state: np.ndarray) -> float:
        """Compute topological constraint satisfaction."""
        return self.check_connectivity(state)
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute gradient of topological constraint."""
        # Numerical gradient
        epsilon = 1e-6
        grad = np.zeros_like(state)
        base_satisfaction = self.satisfaction(state)
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                state[i, j] += epsilon
                grad[i, j] = (self.satisfaction(state) - base_satisfaction) / epsilon
                state[i, j] -= epsilon
        
        return grad


class InformationalConstraint(Constraint):
    """
    Informational constraints based on entropy or mutual information.
    """
    
    def __init__(self,
                 name: str = "informational",
                 precision: float = 1.0,
                 metric: str = "entropy",
                 target_value: Optional[float] = None):
        super().__init__(name, precision)
        self.metric = metric
        self.target_value = target_value
    
    def _compute_entropy(self, state: np.ndarray) -> float:
        """Compute Shannon entropy of state distribution."""
        # Discretize state space
        bins = self.parameters.get("bins", 10)
        hist, _ = np.histogramdd(state.flatten(), bins=bins)
        
        # Normalize to get probabilities
        probs = hist.flatten() / hist.sum()
        probs = probs[probs > 0]  # Remove zeros
        
        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def satisfaction(self, state: np.ndarray) -> float:
        """Compute informational constraint satisfaction."""
        if self.metric == "entropy":
            current_entropy = self._compute_entropy(state)
            if self.target_value is not None:
                # Satisfaction based on distance from target
                diff = abs(current_entropy - self.target_value)
                return 1.0 / (1.0 + diff)
            else:
                # Lower entropy = higher satisfaction
                max_entropy = np.log2(state.size)
                return 1.0 - (current_entropy / max_entropy)
        
        return 0.5  # Default
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute gradient of informational constraint."""
        # Numerical gradient
        epsilon = 1e-6
        grad = np.zeros_like(state)
        base_satisfaction = self.satisfaction(state)
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                state[i, j] += epsilon
                grad[i, j] = (self.satisfaction(state) - base_satisfaction) / epsilon
                state[i, j] -= epsilon
        
        return grad


class BoundaryConstraint(Constraint):
    """
    Boundary constraints that confine the system to specific regions.
    """
    
    def __init__(self,
                 name: str = "boundary",
                 precision: float = 1.0,
                 boundary_type: str = "box",
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(name, precision)
        self.boundary_type = boundary_type
        self.parameters = parameters or {}
    
    def satisfaction(self, state: np.ndarray) -> float:
        """Check if state satisfies boundary constraints."""
        if self.boundary_type == "box":
            bounds = self.parameters.get("bounds", [-10, 10])
            violations = np.sum((state < bounds[0]) | (state > bounds[1]))
            total_elements = state.size
            return 1.0 - (violations / total_elements)
        
        elif self.boundary_type == "sphere":
            center = self.parameters.get("center", np.zeros(state.shape[1]))
            radius = self.parameters.get("radius", 10.0)
            
            distances = np.linalg.norm(state - center, axis=1)
            violations = np.sum(distances > radius)
            return 1.0 - (violations / state.shape[0])
        
        return 0.5  # Default
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute gradient of boundary constraint."""
        # Numerical gradient
        epsilon = 1e-6
        grad = np.zeros_like(state)
        base_satisfaction = self.satisfaction(state)
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                state[i, j] += epsilon
                grad[i, j] = (self.satisfaction(state) - base_satisfaction) / epsilon
                state[i, j] -= epsilon
        
        return grad
