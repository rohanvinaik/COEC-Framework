"""
Constraint base class - Defines satisfaction criteria in COEC framework
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .substrate import Substrate


class Constraint(ABC):
    """
    Single constraint (cᵢ). Returns degree ∈ [0,1] of satisfaction.
    
    Constraints are the fundamental building blocks that shape the
    computational trajectory of COEC systems. They encode both hard
    boundaries and soft preferences through satisfaction degrees.
    
    References:
        - §2.3: Constraints as boundary conditions
        - §2.6: Precision weighting and belief
    """
    
    def __init__(self, precision: float = 1.0, name: Optional[str] = None):
        """
        Initialize constraint with precision weight.
        
        Args:
            precision: Weight/belief precision (§2.6). Higher values indicate
                      stronger constraints. Must be non-negative.
            name: Optional human-readable name for the constraint
                      
        Raises:
            ValueError: If precision is negative
        """
        if precision < 0:
            raise ValueError("Precision must be non-negative")
        self.precision = precision
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def evaluate(self, s: Substrate) -> float:
        """
        Compute satisfaction score for current substrate state.
        
        Args:
            s: Substrate instance to evaluate
            
        Returns:
            Satisfaction degree in [0, 1] where:
            - 0.0 means completely unsatisfied
            - 1.0 means completely satisfied
            - Values in between indicate partial satisfaction
        """
        pass
    
    def grad(self, s: Substrate) -> Optional[Any]:
        """
        Compute gradient of constraint satisfaction w.r.t. substrate state.
        
        This is optional and primarily used for continuous optimization
        approaches. If not implemented, gradient-based evolvers will use
        finite differences or other approximation methods.
        
        Args:
            s: Substrate instance
            
        Returns:
            Gradient in the same shape as substrate state, or None if
            gradient computation is not supported
        """
        return None
    
    def is_satisfied(self, s: Substrate, threshold: float = 0.9) -> bool:
        """
        Check if constraint is satisfied above given threshold.
        
        Args:
            s: Substrate instance to evaluate
            threshold: Satisfaction threshold in [0, 1]
            
        Returns:
            True if satisfaction >= threshold
        """
        return self.evaluate(s) >= threshold
    
    def weighted_satisfaction(self, s: Substrate) -> float:
        """
        Compute precision-weighted satisfaction score.
        
        Args:
            s: Substrate instance
            
        Returns:
            precision * satisfaction
        """
        return self.precision * self.evaluate(s)
    
    def __repr__(self) -> str:
        """String representation of the constraint."""
        return f"{self.name}(precision={self.precision})"