"""
Entropy-Based Compute Governance for COEC.

Implements dynamic resource allocation based on information entropy,
ensuring computational resources are used only where needed. This is core
to the COEC philosophy of avoiding wasted computation on deterministic components.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from ..core.substrate import Substrate
from ..core.system import COECSystem
from ..core.evolution import EvolutionOperator


class ComputeLevel(Enum):
    """Levels of computational intensity."""
    SKIP = "skip"  # Entropy too low, skip processing
    LIGHTWEIGHT = "lightweight"  # Use simple/fast methods
    STANDARD = "standard"  # Normal processing
    INTENSIVE = "intensive"  # Full computational power


@dataclass
class EntropyMetrics:
    """Detailed entropy measurements for decision making."""
    shannon_entropy: float
    spatial_entropy: float  # Entropy across spatial dimensions
    temporal_entropy: float  # Entropy across time
    spectral_entropy: float  # Entropy in frequency domain
    total_entropy: float  # Weighted combination


class EntropyGovernor:
    """
    Core entropy-based compute governance system.
    
    Dynamically allocates computational resources based on the information
    content (entropy) of the data being processed.
    """
    
    def __init__(self,
                 skip_threshold: float = 0.1,
                 lightweight_threshold: float = 0.3,
                 intensive_threshold: float = 0.7):
        self.skip_threshold = skip_threshold
        self.lightweight_threshold = lightweight_threshold
        self.intensive_threshold = intensive_threshold
        
        # Weights for combining different entropy measures
        self.entropy_weights = {
            'shannon': 0.4,
            'spatial': 0.2,
            'temporal': 0.2,
            'spectral': 0.2
        }
        
        # Track statistics for adaptive threshold adjustment
        self.history = []
        self.adaptive_mode = True
    
    def measure_entropy(self, data: np.ndarray) -> EntropyMetrics:
        """
        Compute comprehensive entropy metrics for data.
        
        Args:
            data: Input data array (can be 1D, 2D, or 3D)
            
        Returns:
            EntropyMetrics object with detailed measurements
        """
        # Ensure data is at least 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Shannon entropy
        shannon = self._compute_shannon_entropy(data)
        
        # Spatial entropy (variation across spatial dimensions)
        spatial = self._compute_spatial_entropy(data)
        
        # Temporal entropy (if data has time dimension)
        temporal = self._compute_temporal_entropy(data)
        
        # Spectral entropy (frequency domain)
        spectral = self._compute_spectral_entropy(data)
        
        # Weighted total
        total = (self.entropy_weights['shannon'] * shannon +
                self.entropy_weights['spatial'] * spatial +
                self.entropy_weights['temporal'] * temporal +
                self.entropy_weights['spectral'] * spectral)
        
        return EntropyMetrics(
            shannon_entropy=shannon,
            spatial_entropy=spatial,
            temporal_entropy=temporal,
            spectral_entropy=spectral,
            total_entropy=total
        )
    
    def _compute_shannon_entropy(self, data: np.ndarray) -> float:
        """Compute normalized Shannon entropy."""
        # Flatten and discretize data
        flat_data = data.flatten()
        
        # Use histogram for continuous data
        hist, _ = np.histogram(flat_data, bins=int(np.sqrt(len(flat_data))))
        hist = hist / np.sum(hist)  # Normalize
        
        # Compute entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(hist))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _compute_spatial_entropy(self, data: np.ndarray) -> float:
        """Compute entropy across spatial dimensions."""
        if data.ndim < 2:
            return 0.0
        
        # Compute variance along each spatial dimension
        spatial_vars = []
        for axis in range(data.ndim):
            var = np.var(np.mean(data, axis=axis))
            spatial_vars.append(var)
        
        # Convert variance to entropy-like measure
        total_var = np.sum(spatial_vars)
        if total_var > 0:
            # Use variance ratios as probability distribution
            probs = np.array(spatial_vars) / total_var
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return entropy / np.log(len(spatial_vars))  # Normalize
        return 0.0
    
    def _compute_temporal_entropy(self, data: np.ndarray) -> float:
        """Compute entropy across time dimension (assumed to be first axis)."""
        if data.shape[0] < 2:
            return 0.0
        
        # Compute differences between consecutive time steps
        diffs = np.diff(data, axis=0)
        
        # Measure entropy of changes
        return self._compute_shannon_entropy(diffs)
    
    def _compute_spectral_entropy(self, data: np.ndarray) -> float:
        """Compute entropy in frequency domain."""
        # Compute FFT
        fft_data = np.fft.fft2(data)
        power_spectrum = np.abs(fft_data)**2
        
        # Normalize power spectrum
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        # Compute entropy
        entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(power_spectrum.size)
        return entropy / max_entropy
    
    def should_process(self, data: np.ndarray) -> ComputeLevel:
        """
        Determine appropriate computation level based on entropy.
        
        Returns:
            ComputeLevel indicating how much computation to apply
        """
        metrics = self.measure_entropy(data)
        entropy = metrics.total_entropy
        
        # Store in history for adaptive adjustment
        if self.adaptive_mode:
            self.history.append((entropy, metrics))
            self._adapt_thresholds()
        
        # Determine compute level
        if entropy < self.skip_threshold:
            return ComputeLevel.SKIP
        elif entropy < self.lightweight_threshold:
            return ComputeLevel.LIGHTWEIGHT
        elif entropy < self.intensive_threshold:
            return ComputeLevel.STANDARD
        else:
            return ComputeLevel.INTENSIVE
    
    def _adapt_thresholds(self):
        """Adaptively adjust thresholds based on history."""
        if len(self.history) < 100:
            return
        
        # Get recent entropy values
        recent_entropies = [h[0] for h in self.history[-100:]]
        
        # Compute percentiles for adaptive thresholds
        self.skip_threshold = np.percentile(recent_entropies, 10)
        self.lightweight_threshold = np.percentile(recent_entropies, 40)
        self.intensive_threshold = np.percentile(recent_entropies, 80)
    
    def get_detailed_metrics(self, data: np.ndarray) -> Dict[str, Any]:
        """Get detailed entropy analysis for debugging/visualization."""
        metrics = self.measure_entropy(data)
        compute_level = self.should_process(data)
        
        return {
            'metrics': metrics,
            'compute_level': compute_level,
            'thresholds': {
                'skip': self.skip_threshold,
                'lightweight': self.lightweight_threshold,
                'intensive': self.intensive_threshold
            },
            'recommendation': self._get_recommendation(metrics, compute_level)
        }
    
    def _get_recommendation(self, metrics: EntropyMetrics, level: ComputeLevel) -> str:
        """Generate human-readable recommendation."""
        if level == ComputeLevel.SKIP:
            return "Data has very low entropy. Consider using cached results or simple lookup."
        elif level == ComputeLevel.LIGHTWEIGHT:
            return "Data has low-medium entropy. Use fast approximation methods."
        elif level == ComputeLevel.STANDARD:
            return "Data has medium entropy. Use standard processing pipeline."
        else:
            return "Data has high entropy. Deploy full computational resources."


class AdaptiveComputeManager:
    """
    Manages computational resources across a COEC system based on entropy.
    
    Implements the multi-stage processing pipeline with entropy-based routing.
    """
    
    def __init__(self, base_system: COECSystem):
        self.base_system = base_system
        self.governor = EntropyGovernor()
        
        # Different computational strategies
        self.compute_strategies = {
            ComputeLevel.SKIP: self._skip_compute,
            ComputeLevel.LIGHTWEIGHT: self._lightweight_compute,
            ComputeLevel.STANDARD: self._standard_compute,
            ComputeLevel.INTENSIVE: self._intensive_compute
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_computations': 0,
            'skipped': 0,
            'lightweight': 0,
            'standard': 0,
            'intensive': 0,
            'total_time': 0.0,
            'energy_saved': 0.0
        }
    
    def process(self, input_data: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """
        Process input with entropy-appropriate computation level.
        
        Returns:
            Result and metadata about computation performed
        """
        import time
        start_time = time.time()
        
        # Determine computation level
        compute_level = self.governor.should_process(input_data)
        
        # Update stats
        self.performance_stats['total_computations'] += 1
        self.performance_stats[compute_level.value] += 1
        
        # Execute appropriate strategy
        result = self.compute_strategies[compute_level](input_data)
        
        # Track time and estimate energy
        elapsed_time = time.time() - start_time
        self.performance_stats['total_time'] += elapsed_time
        
        # Estimate energy saved (relative to always using intensive)
        if compute_level != ComputeLevel.INTENSIVE:
            energy_ratio = {
                ComputeLevel.SKIP: 0.01,
                ComputeLevel.LIGHTWEIGHT: 0.1,
                ComputeLevel.STANDARD: 0.5
            }[compute_level]
            self.performance_stats['energy_saved'] += (1 - energy_ratio) * elapsed_time
        
        metadata = {
            'compute_level': compute_level.value,
            'processing_time': elapsed_time,
            'entropy_metrics': self.governor.measure_entropy(input_data)
        }
        
        return result, metadata
    
    def _skip_compute(self, input_data: np.ndarray) -> Any:
        """Skip computation - return cached/default result."""
        # In practice, would check cache or use simple lookup
        # For now, return mean value as trivial result
        return np.mean(input_data)
    
    def _lightweight_compute(self, input_data: np.ndarray) -> Any:
        """Lightweight computation - fast approximations."""
        # Use simple linear operations only
        # Example: PCA-like dimensionality reduction
        mean = np.mean(input_data, axis=0)
        centered = input_data - mean
        
        # Simple projection onto principal component
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        principal_component = vt[0]
        
        projection = np.dot(centered, principal_component)
        return projection
    
    def _standard_compute(self, input_data: np.ndarray) -> Any:
        """Standard computation - normal COEC evolution."""
        # Set substrate state
        self.base_system.substrate.set_state(input_data)
        
        # Run standard evolution (fewer steps)
        result = self.base_system.evolve(steps=100)
        
        return result.final_state
    
    def _intensive_compute(self, input_data: np.ndarray) -> Any:
        """Intensive computation - full COEC with maximum resources."""
        # Set substrate state
        self.base_system.substrate.set_state(input_data)
        
        # Run full evolution with more steps and tighter convergence
        result = self.base_system.evolve(steps=1000)
        
        # Could also add post-processing, ensemble methods, etc.
        return result.final_state
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generate report on computational efficiency gains."""
        total = self.performance_stats['total_computations']
        if total == 0:
            return {'message': 'No computations performed yet'}
        
        return {
            'total_computations': total,
            'distribution': {
                level.value: self.performance_stats[level.value] / total
                for level in ComputeLevel
            },
            'average_time': self.performance_stats['total_time'] / total,
            'energy_saved': self.performance_stats['energy_saved'],
            'efficiency_gain': self.performance_stats['energy_saved'] / 
                              self.performance_stats['total_time']
                              if self.performance_stats['total_time'] > 0 else 0
        }


class EntropyAwareEvolver(EvolutionOperator):
    """
    Evolution operator that adapts its behavior based on local entropy.
    
    Focuses computational effort on high-entropy regions while using
    simple updates in low-entropy regions.
    """
    
    def __init__(self, 
                 base_evolver: EvolutionOperator,
                 entropy_governor: Optional[EntropyGovernor] = None):
        self.base_evolver = base_evolver
        self.entropy_governor = entropy_governor or EntropyGovernor()
        
        # Region-specific evolution parameters
        self.region_params = {
            ComputeLevel.SKIP: {'step_size': 0.0, 'iterations': 0},
            ComputeLevel.LIGHTWEIGHT: {'step_size': 0.1, 'iterations': 1},
            ComputeLevel.STANDARD: {'step_size': 0.01, 'iterations': 5},
            ComputeLevel.INTENSIVE: {'step_size': 0.001, 'iterations': 10}
        }
    
    def evolve(self,
               substrate: Substrate,
               constraints: List[Any],
               energy_landscape: Any,
               steps: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Evolve with entropy-adaptive computation.
        
        Divides substrate into regions and applies different computational
        intensity based on local entropy.
        """
        trajectory = []
        metadata = {
            'entropy_maps': [],
            'compute_distributions': []
        }
        
        for step in range(steps):
            current_state = substrate.get_state()
            trajectory.append(current_state.copy())
            
            # Compute entropy map for current state
            entropy_map = self._compute_entropy_map(current_state)
            metadata['entropy_maps'].append(entropy_map)
            
            # Determine compute level for each region
            compute_map = self._determine_compute_map(entropy_map)
            metadata['compute_distributions'].append(compute_map)
            
            # Apply region-specific evolution
            new_state = self._evolve_with_compute_map(
                current_state, compute_map, constraints, energy_landscape
            )
            
            substrate.set_state(new_state)
        
        return np.array(trajectory), metadata
    
    def _compute_entropy_map(self, state: np.ndarray) -> np.ndarray:
        """Compute local entropy for each region of the state."""
        # Divide state into regions (simplified: each element is a region)
        entropy_map = np.zeros(state.shape[0])
        
        for i in range(state.shape[0]):
            # Compute local entropy (using neighborhood)
            neighborhood_size = min(5, state.shape[0])
            start = max(0, i - neighborhood_size // 2)
            end = min(state.shape[0], i + neighborhood_size // 2 + 1)
            
            local_region = state[start:end]
            metrics = self.entropy_governor.measure_entropy(local_region)
            entropy_map[i] = metrics.total_entropy
        
        return entropy_map
    
    def _determine_compute_map(self, entropy_map: np.ndarray) -> np.ndarray:
        """Determine computation level for each region based on entropy."""
        compute_map = np.zeros(len(entropy_map), dtype=int)
        
        for i, entropy in enumerate(entropy_map):
            if entropy < self.entropy_governor.skip_threshold:
                compute_map[i] = 0  # Skip
            elif entropy < self.entropy_governor.lightweight_threshold:
                compute_map[i] = 1  # Lightweight
            elif entropy < self.entropy_governor.intensive_threshold:
                compute_map[i] = 2  # Standard
            else:
                compute_map[i] = 3  # Intensive
        
        return compute_map
    
    def _evolve_with_compute_map(self, 
                                state: np.ndarray,
                                compute_map: np.ndarray,
                                constraints: List[Any],
                                energy_landscape: Any) -> np.ndarray:
        """Apply different evolution strategies based on compute map."""
        new_state = state.copy()
        
        # Group regions by compute level for batch processing
        for level_idx, level in enumerate(ComputeLevel):
            if level == ComputeLevel.SKIP:
                continue  # No update needed
            
            # Find regions at this compute level
            regions = np.where(compute_map == level_idx)[0]
            if len(regions) == 0:
                continue
            
            # Apply appropriate evolution
            params = self.region_params[level]
            
            for _ in range(params['iterations']):
                # Compute gradients only for these regions
                gradients = self._compute_local_gradients(
                    new_state, regions, constraints, energy_landscape
                )
                
                # Update with appropriate step size
                new_state[regions] -= params['step_size'] * gradients
        
        return new_state
    
    def _compute_local_gradients(self,
                                state: np.ndarray,
                                regions: np.ndarray,
                                constraints: List[Any],
                                energy_landscape: Any) -> np.ndarray:
        """Compute gradients only for specified regions."""
        # Simplified - in practice would use constraint/energy gradients
        local_state = state[regions]
        
        # Random gradient for demonstration
        gradients = np.random.randn(*local_state.shape) * 0.1
        
        return gradients
