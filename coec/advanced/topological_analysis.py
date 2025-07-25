"""
Topological Data Analysis for COEC.

Implements persistent homology, mapper algorithm, and topological feature
extraction for understanding and optimizing COEC systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import warnings

from ..core.constraint import Constraint
from ..core.substrate import Substrate


@dataclass
class PersistencePair:
    """Birth-death pair in persistent homology."""
    dimension: int  # 0=components, 1=loops, 2=voids
    birth: float
    death: float
    
    @property
    def persistence(self) -> float:
        """Lifetime of the topological feature."""
        return self.death - self.birth
    
    @property
    def midpoint(self) -> float:
        """Midpoint of the feature's lifetime."""
        return (self.birth + self.death) / 2


class SimplexTree:
    """
    Simplified simplex tree for computing persistent homology.
    
    Note: This is a basic implementation for demonstration.
    In practice, would use GUDHI or Dionysus libraries.
    """
    
    def __init__(self):
        self.simplices = defaultdict(set)  # dimension -> set of simplices
        self.filtration_values = {}
    
    def add_simplex(self, vertices: Tuple[int, ...], filtration: float):
        """Add simplex with given filtration value."""
        dimension = len(vertices) - 1
        self.simplices[dimension].add(vertices)
        self.filtration_values[vertices] = filtration
        
        # Add all faces
        if dimension > 0:
            for i in range(len(vertices)):
                face = tuple(v for j, v in enumerate(vertices) if j != i)
                if face not in self.filtration_values:
                    self.add_simplex(face, filtration)
    
    def build_from_distance_matrix(self, distances: np.ndarray, max_dimension: int = 2):
        """Build Vietoris-Rips complex from distance matrix."""
        n_points = distances.shape[0]
        
        # Add vertices (0-simplices)
        for i in range(n_points):
            self.add_simplex((i,), 0.0)
        
        # Add edges (1-simplices)
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                edges.append((distances[i, j], (i, j)))
        
        # Sort by distance
        edges.sort()
        
        # Add edges in order
        for dist, (i, j) in edges:
            self.add_simplex((i, j), dist)
        
        # Add higher dimensional simplices if requested
        if max_dimension >= 2:
            # Add triangles (2-simplices)
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    for k in range(j + 1, n_points):
                        # Filtration value is maximum edge length
                        filt_value = max(distances[i, j], distances[i, k], distances[j, k])
                        self.add_simplex((i, j, k), filt_value)


class PersistentHomology:
    """
    Compute persistent homology of point clouds and functions.
    
    Tracks topological features (components, loops, voids) across scales.
    """
    
    def __init__(self):
        self.persistence_pairs = []
    
    def compute_persistence(self, points: np.ndarray, 
                          max_dimension: int = 1) -> List[PersistencePair]:
        """
        Compute persistent homology of point cloud.
        
        Args:
            points: (n_points, n_dims) array
            max_dimension: Maximum homological dimension to compute
            
        Returns:
            List of persistence pairs
        """
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points))
        
        # Build simplex tree
        tree = SimplexTree()
        tree.build_from_distance_matrix(distances, max_dimension)
        
        # Compute persistence (simplified - just for demonstration)
        # In practice, would use proper algorithm
        pairs = []
        
        # 0-dimensional features (connected components)
        # All points are born at 0, components die when merged
        n_points = len(points)
        component_deaths = self._compute_component_deaths(distances)
        
        for death in component_deaths[:-1]:  # Last component never dies
            pairs.append(PersistencePair(0, 0.0, death))
        
        # Add infinite persistence for last component
        pairs.append(PersistencePair(0, 0.0, float('inf')))
        
        # 1-dimensional features (loops) - simplified
        if max_dimension >= 1:
            loop_pairs = self._detect_loops_simplified(distances)
            pairs.extend(loop_pairs)
        
        self.persistence_pairs = pairs
        return pairs
    
    def _compute_component_deaths(self, distances: np.ndarray) -> List[float]:
        """Compute when connected components merge (using MST)."""
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        # Get MST
        mst = minimum_spanning_tree(distances)
        
        # Edge weights in MST tell us when components merge
        edges = []
        for i in range(distances.shape[0]):
            for j in range(i + 1, distances.shape[1]):
                if mst[i, j] > 0 or mst[j, i] > 0:
                    edges.append(max(mst[i, j], mst[j, i]))
        
        return sorted(edges)
    
    def _detect_loops_simplified(self, distances: np.ndarray) -> List[PersistencePair]:
        """Simplified loop detection based on cycles in graph."""
        pairs = []
        n_points = distances.shape[0]
        
        # Very simplified: look for triangles and their birth/death
        for i in range(n_points):
            for j in range(i + 1, n_points):
                for k in range(j + 1, n_points):
                    # Triangle edges
                    edges = [distances[i, j], distances[i, k], distances[j, k]]
                    
                    # Birth = when last edge appears
                    birth = max(edges)
                    
                    # Death = simplified heuristic (when filled in)
                    # In real implementation, would track when loop is filled
                    death = birth * 1.5
                    
                    if death < float('inf'):
                        pairs.append(PersistencePair(1, birth, death))
        
        return pairs
    
    def persistence_diagram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get birth-death coordinates for persistence diagram.
        
        Returns:
            births: Array of birth times
            deaths: Array of death times
        """
        births = np.array([p.birth for p in self.persistence_pairs])
        deaths = np.array([p.death for p in self.persistence_pairs])
        
        return births, deaths
    
    def persistence_landscape(self, resolution: int = 100) -> np.ndarray:
        """
        Compute persistence landscape (stable summary of persistence diagram).
        
        Returns discretized landscape functions.
        """
        # Get finite persistence pairs
        finite_pairs = [p for p in self.persistence_pairs if p.death < float('inf')]
        
        if not finite_pairs:
            return np.zeros((1, resolution))
        
        # Determine range
        max_val = max(max(p.birth, p.death) for p in finite_pairs)
        t_values = np.linspace(0, max_val, resolution)
        
        # Compute landscape functions
        landscapes = []
        
        for k in range(min(5, len(finite_pairs))):  # First 5 landscape functions
            landscape_k = np.zeros(resolution)
            
            # For each time t, find k-th largest persistence
            for i, t in enumerate(t_values):
                values = []
                for pair in finite_pairs:
                    if pair.birth <= t <= pair.death:
                        # Height of tent function at t
                        height = min(t - pair.birth, pair.death - t)
                        values.append(height)
                
                if len(values) > k:
                    values.sort(reverse=True)
                    landscape_k[i] = values[k]
            
            landscapes.append(landscape_k)
        
        return np.array(landscapes)


class TopologicalFeatureExtractor:
    """
    Extract topological features from COEC substrates for analysis.
    
    These features can be used for anomaly detection, compression guidance,
    or as additional constraints.
    """
    
    def __init__(self):
        self.homology = PersistentHomology()
    
    def extract_features(self, substrate: Substrate) -> Dict[str, Any]:
        """
        Extract comprehensive topological features from substrate state.
        
        Returns dictionary of features.
        """
        state = substrate.get_state()
        
        features = {}
        
        # Compute persistent homology
        persistence_pairs = self.homology.compute_persistence(state)
        
        # Basic statistics
        features['num_components'] = sum(1 for p in persistence_pairs if p.dimension == 0)
        features['num_loops'] = sum(1 for p in persistence_pairs if p.dimension == 1)
        
        # Persistence statistics by dimension
        for dim in range(2):
            dim_pairs = [p for p in persistence_pairs if p.dimension == dim]
            if dim_pairs:
                persistences = [p.persistence for p in dim_pairs if p.death < float('inf')]
                if persistences:
                    features[f'dim{dim}_mean_persistence'] = np.mean(persistences)
                    features[f'dim{dim}_max_persistence'] = np.max(persistences)
                    features[f'dim{dim}_total_persistence'] = np.sum(persistences)
        
        # Compute Betti numbers at different scales
        scales = np.linspace(0, 1, 10)
        for i, scale in enumerate(scales):
            betti = self._compute_betti_numbers(persistence_pairs, scale)
            for dim, count in enumerate(betti):
                features[f'betti_{dim}_at_scale_{i}'] = count
        
        # Persistence entropy
        features['persistence_entropy'] = self._compute_persistence_entropy(persistence_pairs)
        
        # Wasserstein distance to baseline (if available)
        # features['wasserstein_distance'] = self._compute_wasserstein_distance(pairs, baseline)
        
        return features
    
    def _compute_betti_numbers(self, pairs: List[PersistencePair], scale: float) -> List[int]:
        """Compute Betti numbers at given scale."""
        betti = [0, 0, 0]  # dimensions 0, 1, 2
        
        for pair in pairs:
            if pair.birth <= scale < pair.death:
                betti[pair.dimension] += 1
        
        return betti
    
    def _compute_persistence_entropy(self, pairs: List[PersistencePair]) -> float:
        """Compute entropy of persistence diagram."""
        # Get all finite persistences
        persistences = [p.persistence for p in pairs if p.death < float('inf')]
        
        if not persistences:
            return 0.0
        
        # Normalize to probability distribution
        total = sum(persistences)
        if total == 0:
            return 0.0
        
        probs = np.array(persistences) / total
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy
    
    def detect_topological_anomalies(self, 
                                   substrate: Substrate,
                                   reference_features: Dict[str, Any],
                                   threshold: float = 3.0) -> List[str]:
        """
        Detect topological anomalies compared to reference.
        
        Returns list of anomalous features.
        """
        current_features = self.extract_features(substrate)
        
        anomalies = []
        
        for feature_name, ref_value in reference_features.items():
            if feature_name in current_features:
                current_value = current_features[feature_name]
                
                # Compute z-score
                if isinstance(ref_value, dict) and 'mean' in ref_value and 'std' in ref_value:
                    z_score = abs(current_value - ref_value['mean']) / (ref_value['std'] + 1e-10)
                    
                    if z_score > threshold:
                        anomalies.append(f"{feature_name}: z-score={z_score:.2f}")
        
        return anomalies


class PersistentHomologyConstraint(Constraint):
    """
    Constraint based on topological features.
    
    Ensures substrate maintains certain topological properties
    (e.g., connectedness, absence of large voids).
    """
    
    def __init__(self,
                 name: str,
                 target_betti: Optional[List[int]] = None,
                 max_persistence: Optional[Dict[int, float]] = None,
                 min_persistence: Optional[Dict[int, float]] = None):
        super().__init__(name)
        
        self.target_betti = target_betti or [1, 0, 0]  # Default: connected, no loops
        self.max_persistence = max_persistence or {0: 1.0, 1: 0.5}
        self.min_persistence = min_persistence or {}
        
        self.homology = PersistentHomology()
    
    def satisfaction(self, state: np.ndarray) -> float:
        """
        Compute satisfaction based on topological features.
        
        Returns value in [0, 1].
        """
        # Compute persistent homology
        pairs = self.homology.compute_persistence(state, max_dimension=len(self.target_betti)-1)
        
        satisfaction_scores = []
        
        # Check Betti numbers
        if self.target_betti:
            betti = self._compute_current_betti(pairs)
            betti_score = 0.0
            
            for dim, (target, current) in enumerate(zip(self.target_betti, betti)):
                if target == current:
                    betti_score += 1.0
                else:
                    # Partial credit based on distance
                    betti_score += np.exp(-abs(target - current))
            
            betti_score /= len(self.target_betti)
            satisfaction_scores.append(betti_score)
        
        # Check persistence bounds
        persistence_score = 1.0
        
        for pair in pairs:
            if pair.death == float('inf'):
                continue
            
            dim = pair.dimension
            persistence = pair.persistence
            
            # Check maximum persistence
            if dim in self.max_persistence:
                if persistence > self.max_persistence[dim]:
                    persistence_score *= np.exp(-(persistence - self.max_persistence[dim]))
            
            # Check minimum persistence  
            if dim in self.min_persistence:
                if persistence < self.min_persistence[dim]:
                    persistence_score *= np.exp(-(self.min_persistence[dim] - persistence))
        
        satisfaction_scores.append(persistence_score)
        
        # Average all scores
        return np.mean(satisfaction_scores)
    
    def _compute_current_betti(self, pairs: List[PersistencePair]) -> List[int]:
        """Compute current Betti numbers."""
        # Count features that persist at scale 0.5 (midpoint)
        scale = 0.5
        betti = [0] * len(self.target_betti)
        
        for pair in pairs:
            if pair.dimension < len(betti) and pair.birth <= scale < pair.death:
                betti[pair.dimension] += 1
        
        return betti
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gradient to guide toward topological targets.
        
        Note: Topological gradients are challenging - this is simplified.
        """
        epsilon = 1e-5
        grad = np.zeros_like(state)
        base_satisfaction = self.satisfaction(state)
        
        # Finite differences with special handling for topology
        for i in range(state.shape[0]):
            # Move point in different directions
            for j in range(state.shape[1]):
                state[i, j] += epsilon
                perturbed_satisfaction = self.satisfaction(state)
                
                # Topological changes are discrete, so amplify gradient
                if abs(perturbed_satisfaction - base_satisfaction) > 1e-10:
                    grad[i, j] = 10 * (perturbed_satisfaction - base_satisfaction) / epsilon
                
                state[i, j] -= epsilon
        
        return grad


class TopologicalCompressor:
    """
    Use topological features to guide compression.
    
    Identifies compressible substructures based on topological analysis.
    """
    
    def __init__(self, persistence_threshold: float = 0.1):
        self.persistence_threshold = persistence_threshold
        self.feature_extractor = TopologicalFeatureExtractor()
    
    def identify_compressible_regions(self, substrate: Substrate) -> Dict[str, Any]:
        """
        Identify regions that can be compressed without losing topology.
        
        Returns compression guidance.
        """
        state = substrate.get_state()
        
        # Extract topological features
        features = self.feature_extractor.extract_features(substrate)
        
        # Compute persistent homology
        pairs = self.feature_extractor.homology.persistence_pairs
        
        # Identify low-persistence features (noise)
        noise_features = [
            p for p in pairs 
            if p.death < float('inf') and p.persistence < self.persistence_threshold
        ]
        
        compression_info = {
            'compressible_features': len(noise_features),
            'total_features': len(pairs),
            'compression_ratio': len(noise_features) / len(pairs) if pairs else 0,
            'recommended_dimensions': self._recommend_dimensions(features),
            'noise_threshold': self.persistence_threshold
        }
        
        return compression_info
    
    def _recommend_dimensions(self, features: Dict[str, Any]) -> int:
        """Recommend compressed dimension based on topology."""
        # Use persistence entropy as guide
        entropy = features.get('persistence_entropy', 1.0)
        
        # Higher entropy = more complex topology = need more dimensions
        # This is a heuristic
        base_dim = 10
        recommended = int(base_dim * (1 + entropy))
        
        return recommended
