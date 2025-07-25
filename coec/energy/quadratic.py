"""
Quadratic energy landscape - E(x) = x^T Q x
"""
import numpy as np
from typing import Optional, Union

from ..core.energy import EnergyLandscape
from ..core.substrate import Substrate
from ..substrates.euclidean import EuclideanSubstrate


class QuadraticEnergy(EnergyLandscape):
    """
    Quadratic energy function: E(x) = ½ x^T Q x + b^T x + c.
    
    This is one of the simplest energy landscapes, creating a
    paraboloid in state space. The matrix Q determines the shape:
    - Positive definite Q: unique global minimum
    - Positive semi-definite Q: potentially flat directions
    - Indefinite Q: saddle points
    
    Useful for:
    - Harmonic oscillators
    - Elastic networks
    - Quadratic optimization
    - Testing and validation
    """
    
    def __init__(
        self,
        Q: Union[np.ndarray, None] = None,
        b: Optional[np.ndarray] = None,
        c: float = 0.0,
        name: Optional[str] = None
    ):
        """
        Initialize quadratic energy landscape.
        
        Args:
            Q: Quadratic form matrix (should be symmetric).
               If None, defaults to identity matrix.
            b: Linear term vector. If None, defaults to zero.
            c: Constant term.
            name: Optional name for the energy function.
        """
        super().__init__(name)
        
        self.Q = Q
        self.b = b
        self.c = c
        self._dimension = None
        
        # Validate and symmetrize Q if provided
        if Q is not None:
            Q = np.asarray(Q, dtype=np.float64)
            if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
                raise ValueError("Q must be a square matrix")
            # Ensure symmetry
            self.Q = 0.5 * (Q + Q.T)
            self._dimension = Q.shape[0]
        
        # Validate b if provided
        if b is not None:
            b = np.asarray(b, dtype=np.float64)
            if b.ndim != 1:
                raise ValueError("b must be a 1D vector")
            self.b = b
            
            if self._dimension is None:
                self._dimension = len(b)
            elif len(b) != self._dimension:
                raise ValueError("Dimension mismatch between Q and b")
    
    def _ensure_initialized(self, dim: int) -> None:
        """Ensure Q and b are initialized for given dimension."""
        if self.Q is None:
            self.Q = np.eye(dim, dtype=np.float64)
        
        if self.b is None:
            self.b = np.zeros(dim, dtype=np.float64)
    
    def energy(self, s: Substrate) -> float:
        """
        Compute quadratic energy E(x) = ½ x^T Q x + b^T x + c.
        
        Args:
            s: Substrate instance (must be EuclideanSubstrate)
            
        Returns:
            Energy value
            
        Raises:
            TypeError: If substrate is not EuclideanSubstrate
        """
        if not isinstance(s, EuclideanSubstrate):
            raise TypeError(
                f"QuadraticEnergy requires EuclideanSubstrate, got {type(s)}"
            )
        
        x = s.state
        dim = s.dimension()
        self._ensure_initialized(dim)
        
        # Compute ½ x^T Q x + b^T x + c
        quadratic_term = 0.5 * np.dot(x, np.dot(self.Q, x))
        linear_term = np.dot(self.b, x)
        
        return float(quadratic_term + linear_term + self.c)
    
    def grad(self, s: Substrate) -> np.ndarray:
        """
        Compute gradient ∇E = Qx + b.
        
        For symmetric Q, the gradient is exactly Qx + b.
        
        Args:
            s: Substrate instance
            
        Returns:
            Gradient vector
        """
        if not isinstance(s, EuclideanSubstrate):
            raise TypeError(
                f"QuadraticEnergy requires EuclideanSubstrate, got {type(s)}"
            )
        
        x = s.state
        dim = s.dimension()
        self._ensure_initialized(dim)
        
        return np.dot(self.Q, x) + self.b
    
    def hessian(self, s: Substrate) -> np.ndarray:
        """
        Return Hessian matrix (which is just Q for quadratic energy).
        
        Args:
            s: Substrate instance
            
        Returns:
            Hessian matrix Q
        """
        if not isinstance(s, EuclideanSubstrate):
            raise TypeError(
                f"QuadraticEnergy requires EuclideanSubstrate, got {type(s)}"
            )
        
        dim = s.dimension()
        self._ensure_initialized(dim)
        
        return self.Q.copy()
    
    def minimum(self) -> Optional[np.ndarray]:
        """
        Compute the global minimum of the quadratic energy.
        
        For E(x) = ½ x^T Q x + b^T x + c, the minimum is at x* = -Q^(-1)b
        if Q is positive definite.
        
        Returns:
            Minimum point if Q is invertible, None otherwise
        """
        if self.Q is None or self.b is None:
            return None
        
        try:
            # Solve Qx = -b
            x_min = np.linalg.solve(self.Q, -self.b)
            return x_min
        except np.linalg.LinAlgError:
            # Q is singular
            return None
    
    def is_positive_definite(self) -> bool:
        """
        Check if the quadratic form is positive definite.
        
        Returns:
            True if all eigenvalues of Q are positive
        """
        if self.Q is None:
            return True  # Identity matrix is positive definite
        
        try:
            # Check via Cholesky decomposition
            np.linalg.cholesky(self.Q)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def eigenvalues(self) -> Optional[np.ndarray]:
        """
        Compute eigenvalues of Q.
        
        Returns:
            Sorted eigenvalues (smallest to largest) or None
        """
        if self.Q is None:
            return None
        
        eigvals = np.linalg.eigvalsh(self.Q)
        return np.sort(eigvals)
    
    def condition_number(self) -> Optional[float]:
        """
        Compute condition number of Q (ratio of largest to smallest eigenvalue).
        
        Returns:
            Condition number or None if Q is not set
        """
        eigvals = self.eigenvalues()
        if eigvals is None or len(eigvals) == 0:
            return None
        
        min_eig = np.abs(eigvals[0])
        max_eig = np.abs(eigvals[-1])
        
        if min_eig < 1e-15:
            return np.inf
        
        return max_eig / min_eig
    
    def __repr__(self) -> str:
        """String representation of the energy landscape."""
        if self.Q is not None:
            cond = self.condition_number()
            pd = self.is_positive_definite()
            return f"QuadraticEnergy(dim={self.Q.shape[0]}, positive_def={pd}, cond={cond:.2e})"
        else:
            return "QuadraticEnergy(uninitialized)"