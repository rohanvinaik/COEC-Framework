"""
Euclidean substrate - ℝⁿ space implementation
"""
from __future__ import annotations

import numpy as np
from typing import Union

from ..core.substrate import Substrate


class EuclideanSubstrate(Substrate['EuclideanSubstrate']):
    """
    Toy ℝⁿ space – good for optimization & unit tests.
    
    This is the simplest substrate type, representing points in
    n-dimensional Euclidean space. Useful for:
    - Optimization problems
    - Testing and validation
    - Simple physical systems
    - Neural network weight spaces
    """
    
    def __init__(self, state: Union[np.ndarray, list]):
        """
        Initialize Euclidean substrate.
        
        Args:
            state: Initial state vector (numpy array or list)
        """
        if isinstance(state, list):
            state = np.array(state, dtype=np.float64)
        else:
            state = np.asarray(state, dtype=np.float64)
        
        if state.ndim != 1:
            raise ValueError("EuclideanSubstrate requires 1D state vector")
        
        super().__init__(state)
    
    def _clone_state(self) -> np.ndarray:
        """Create a deep copy of the state array."""
        return self.state.copy()
    
    def distance(self, other: EuclideanSubstrate) -> float:
        """
        Compute L2 (Euclidean) distance to another substrate.
        
        Args:
            other: Another EuclideanSubstrate instance
            
        Returns:
            L2 distance between states
            
        Raises:
            TypeError: If other is not EuclideanSubstrate
            ValueError: If dimensions don't match
        """
        if not isinstance(other, EuclideanSubstrate):
            raise TypeError(f"Cannot compute distance to {type(other)}")
        
        if self.state.shape != other.state.shape:
            raise ValueError(
                f"Dimension mismatch: {self.state.shape} vs {other.state.shape}"
            )
        
        return float(np.linalg.norm(self.state - other.state))
    
    def dimension(self) -> int:
        """Return the dimensionality of the state space."""
        return self.state.shape[0]
    
    def norm(self) -> float:
        """Compute L2 norm of the state vector."""
        return float(np.linalg.norm(self.state))
    
    def dot(self, other: EuclideanSubstrate) -> float:
        """
        Compute dot product with another substrate.
        
        Args:
            other: Another EuclideanSubstrate instance
            
        Returns:
            Dot product of state vectors
        """
        if not isinstance(other, EuclideanSubstrate):
            raise TypeError(f"Cannot compute dot product with {type(other)}")
        
        return float(np.dot(self.state, other.state))
    
    def __add__(self, other: EuclideanSubstrate) -> EuclideanSubstrate:
        """Add two substrates element-wise."""
        if not isinstance(other, EuclideanSubstrate):
            raise TypeError(f"Cannot add {type(other)} to EuclideanSubstrate")
        
        return EuclideanSubstrate(self.state + other.state)
    
    def __sub__(self, other: EuclideanSubstrate) -> EuclideanSubstrate:
        """Subtract two substrates element-wise."""
        if not isinstance(other, EuclideanSubstrate):
            raise TypeError(f"Cannot subtract {type(other)} from EuclideanSubstrate")
        
        return EuclideanSubstrate(self.state - other.state)
    
    def __mul__(self, scalar: float) -> EuclideanSubstrate:
        """Multiply substrate by scalar."""
        return EuclideanSubstrate(self.state * scalar)
    
    def __rmul__(self, scalar: float) -> EuclideanSubstrate:
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        """String representation with state preview."""
        if self.dimension() <= 4:
            state_str = str(self.state)
        else:
            # Show first 3 and last element for large vectors
            state_str = f"[{self.state[0]:.3f}, {self.state[1]:.3f}, {self.state[2]:.3f}, ..., {self.state[-1]:.3f}]"
        
        return f"EuclideanSubstrate(dim={self.dimension()}, state={state_str})"