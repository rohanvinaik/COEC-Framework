"""
Substrate base class - State-bearing medium (S) in COEC framework
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

S = TypeVar('S', bound='Substrate')


class Substrate(ABC, Generic[S]):
    """
    State-bearing medium (S). Implements a finite/high-dimensional state-space.
    
    This is the fundamental data structure that holds the system's configuration
    and provides methods for state manipulation and distance computation.
    
    References:
        - §2.1: Substrate as part of the 7-tuple ontology
        - §2.2: Configuration space and substrates
    """
    
    def __init__(self, state: Any):
        """
        Initialize substrate with given state.
        
        Args:
            state: The initial state configuration. Shape and dtype are 
                   substrate-specific.
        """
        self.state = state
    
    def clone(self: S) -> S:
        """
        Create a deep copy of the current substrate.
        
        Returns:
            A new substrate instance with cloned state
        """
        return self.__class__(state=self._clone_state())
    
    @abstractmethod
    def _clone_state(self) -> Any:
        """
        Create a deep copy of the internal state.
        
        This method must be implemented by concrete substrate classes
        to handle their specific state representations.
        
        Returns:
            A deep copy of the state
        """
        pass
    
    @abstractmethod
    def distance(self, other: S) -> float:
        """
        Compute metric distance between this substrate and another.
        
        This defines the metric on Ω_S. Defaults to L2 distance for 
        Euclidean substrates but may be graph distance, manifold distance,
        or other domain-specific metrics.
        
        Args:
            other: Another substrate instance of the same type
            
        Returns:
            Non-negative distance value
            
        Raises:
            TypeError: If other is not of compatible substrate type
        """
        pass
    
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimensionality of the substrate's state space.
        
        Returns:
            Integer dimension of the state space
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the substrate."""
        return f"{self.__class__.__name__}(dim={self.dimension()})"