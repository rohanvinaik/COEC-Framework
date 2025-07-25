"""
Energy landscape base class - Combined physical + informational potential
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .substrate import Substrate


class EnergyLandscape(ABC):
    """
    Combined physical + informational potential (E).
    
    The energy landscape defines the natural dynamics of the system in the
    absence of constraints. It can represent physical potentials (e.g.,
    molecular forces), informational measures (e.g., entropy), or composite
    objectives combining multiple factors.
    
    References:
        - §2.4: Energy landscapes and combined dynamics
        - §4.1: Information-theoretic energy formulations
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize energy landscape.
        
        Args:
            name: Optional human-readable name for the energy function
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def energy(self, s: Substrate) -> float:
        """
        Compute energy value for given substrate state.
        
        Lower energy values typically indicate more favorable states,
        following physical convention. The energy scale is relative and
        system-specific.
        
        Args:
            s: Substrate instance
            
        Returns:
            Energy value (scalar)
        """
        pass
    
    def grad(self, s: Substrate) -> Optional[Any]:
        """
        Compute gradient of energy w.r.t. substrate state.
        
        This is optional but highly recommended for gradient-based
        evolution operators. If not implemented, numerical gradients
        may be used at computational cost.
        
        Args:
            s: Substrate instance
            
        Returns:
            Gradient in the same shape as substrate state, or None if
            gradient computation is not supported
        """
        return None
    
    def hessian(self, s: Substrate) -> Optional[Any]:
        """
        Compute Hessian (second derivative) of energy.
        
        Used for advanced optimization methods and stability analysis.
        Most implementations will return None.
        
        Args:
            s: Substrate instance
            
        Returns:
            Hessian matrix or None if not supported
        """
        return None
    
    def is_local_minimum(self, s: Substrate, tolerance: float = 1e-6) -> bool:
        """
        Check if substrate is at a local minimum (approximately).
        
        Args:
            s: Substrate instance
            tolerance: Gradient magnitude tolerance
            
        Returns:
            True if gradient magnitude is below tolerance
        """
        grad = self.grad(s)
        if grad is None:
            return False
        
        # Compute gradient magnitude (implementation depends on substrate type)
        import numpy as np
        grad_magnitude = np.linalg.norm(grad.flatten())
        return grad_magnitude < tolerance
    
    def __repr__(self) -> str:
        """String representation of the energy landscape."""
        return f"{self.name}()"