"""
Linear constraint implementation - c(x) = σ(w·x - b)
"""
import numpy as np
from typing import Optional, Union

from ..core.constraint import Constraint
from ..core.substrate import Substrate
from ..substrates.euclidean import EuclideanSubstrate


class LinearConstraint(Constraint):
    """
    Linear constraint with sigmoid satisfaction: c(x) = σ(w·x - b).
    
    This constraint type is satisfied when the linear combination w·x
    exceeds threshold b. The sigmoid function provides smooth gradients
    for optimization.
    
    Good for:
    - Hyperplane boundaries
    - Resource limits
    - Linear inequalities
    - Quick demos and unit tests
    """
    
    def __init__(
        self,
        w: Union[np.ndarray, list],
        b: float,
        precision: float = 1.0,
        temperature: float = 1.0,
        name: Optional[str] = None
    ):
        """
        Initialize linear constraint.
        
        Args:
            w: Weight vector defining hyperplane normal
            b: Bias/threshold value
            precision: Constraint weight/importance
            temperature: Sigmoid temperature (higher = sharper transition)
            name: Optional constraint name
        """
        super().__init__(precision, name)
        
        if isinstance(w, list):
            w = np.array(w, dtype=np.float64)
        else:
            w = np.asarray(w, dtype=np.float64)
        
        self.w = w
        self.b = b
        self.temperature = temperature
    
    def evaluate(self, s: Substrate) -> float:
        """
        Compute sigmoid satisfaction for linear constraint.
        
        Args:
            s: Substrate instance (must be EuclideanSubstrate)
            
        Returns:
            Satisfaction in [0, 1] where high values mean w·x > b
            
        Raises:
            TypeError: If substrate is not EuclideanSubstrate
        """
        if not isinstance(s, EuclideanSubstrate):
            raise TypeError(
                f"LinearConstraint requires EuclideanSubstrate, got {type(s)}"
            )
        
        # Check dimension compatibility
        if len(self.w) != s.dimension():
            raise ValueError(
                f"Weight dimension {len(self.w)} doesn't match "
                f"substrate dimension {s.dimension()}"
            )
        
        # Compute w·x - b
        activation = np.dot(self.w, s.state) - self.b
        
        # Apply sigmoid with temperature
        return float(1.0 / (1.0 + np.exp(-activation / self.temperature)))
    
    def grad(self, s: Substrate) -> np.ndarray:
        """
        Compute gradient of constraint satisfaction.
        
        The gradient of σ(z) is σ(z)(1 - σ(z)) · ∂z/∂x = σ(z)(1 - σ(z)) · w
        
        Args:
            s: Substrate instance
            
        Returns:
            Gradient vector
        """
        if not isinstance(s, EuclideanSubstrate):
            raise TypeError(
                f"LinearConstraint requires EuclideanSubstrate, got {type(s)}"
            )
        
        # Current satisfaction
        σ = self.evaluate(s)
        
        # Gradient of sigmoid
        return (σ * (1 - σ) / self.temperature) * self.w
    
    def get_violation_direction(self, s: EuclideanSubstrate) -> np.ndarray:
        """
        Get direction to move to better satisfy constraint.
        
        Args:
            s: Current substrate state
            
        Returns:
            Unit vector pointing toward constraint satisfaction
        """
        # Move in direction of positive gradient
        grad = self.grad(s)
        norm = np.linalg.norm(grad)
        
        if norm > 1e-10:
            return grad / norm
        else:
            return np.zeros_like(grad)
    
    def distance_to_satisfaction(self, s: EuclideanSubstrate) -> float:
        """
        Compute distance to constraint boundary (w·x = b).
        
        Args:
            s: Current substrate state
            
        Returns:
            Signed distance (positive if satisfied, negative if violated)
        """
        w_norm = np.linalg.norm(self.w)
        if w_norm < 1e-10:
            return 0.0
        
        return (np.dot(self.w, s.state) - self.b) / w_norm
    
    def project_to_boundary(self, s: EuclideanSubstrate) -> EuclideanSubstrate:
        """
        Project state onto constraint boundary hyperplane.
        
        Args:
            s: Current substrate state
            
        Returns:
            New substrate at closest point on w·x = b
        """
        w_norm_sq = np.dot(self.w, self.w)
        if w_norm_sq < 1e-10:
            return s.clone()
        
        # Project onto hyperplane
        t = (self.b - np.dot(self.w, s.state)) / w_norm_sq
        projected_state = s.state + t * self.w
        
        return EuclideanSubstrate(projected_state)
    
    def __repr__(self) -> str:
        """String representation of the constraint."""
        return (
            f"LinearConstraint(dim={len(self.w)}, "
            f"precision={self.precision}, "
            f"temp={self.temperature})"
        )