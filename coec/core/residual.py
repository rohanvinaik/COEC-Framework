"""
Residual function protocol - Maps substrate states to computational outputs
"""
from typing import Protocol, Any, runtime_checkable

from .substrate import Substrate


@runtime_checkable
class ResidualFunction(Protocol):
    """
    Maps observable substrate state(s) to computational output (R).
    
    The residual function extracts the "answer" or "result" from the
    evolved substrate state. This is what makes COEC a computational
    framework rather than just a dynamical system.
    
    References:
        - §2.7: Residual functions and computational output
        - §3: Different residual patterns for COEC classes
    """
    
    def __call__(self, s: Substrate) -> Any:
        """
        Extract computational output from substrate state.
        
        Args:
            s: Evolved substrate instance
            
        Returns:
            Computational output (type depends on specific problem)
        """
        ...