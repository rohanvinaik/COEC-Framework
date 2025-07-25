"""
Distributed and Federated COEC Systems.

Implements privacy-preserving distributed computation, federated learning,
and Byzantine fault-tolerant consensus for COEC systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
from collections import defaultdict

from ..core.system import COECSystem, COECResult
from ..core.substrate import Substrate
from ..core.constraint import Constraint


@dataclass
class ParticipantUpdate:
    """Update from a single participant in federated system."""
    participant_id: str
    substrate_delta: np.ndarray
    constraint_satisfactions: Dict[str, float]
    local_energy: float
    metadata: Dict[str, Any]


class PrivacyPreservingTransform:
    """
    Implements privacy-preserving transformations for distributed COEC.
    
    Includes differential privacy, secure aggregation, and obfuscation.
    """
    
    def __init__(self, privacy_level: str = "moderate"):
        self.privacy_level = privacy_level
        
        # Privacy parameters
        self.privacy_params = {
            "low": {"epsilon": 10.0, "delta": 1e-5, "noise_scale": 0.01},
            "moderate": {"epsilon": 1.0, "delta": 1e-6, "noise_scale": 0.05},
            "high": {"epsilon": 0.1, "delta": 1e-7, "noise_scale": 0.1}
        }[privacy_level]
        
        # For secure aggregation
        self.aggregation_threshold = 3  # Minimum participants for aggregation
    
    def add_differential_privacy(self, data: np.ndarray) -> np.ndarray:
        """
        Add calibrated noise for differential privacy.
        
        Uses Laplace mechanism for bounded sensitivity.
        """
        epsilon = self.privacy_params["epsilon"]
        
        # Compute sensitivity (max change from single record)
        sensitivity = 2.0 / data.size  # Normalized
        
        # Laplace noise scale
        noise_scale = sensitivity / epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, noise_scale, data.shape)
        
        return data + noise
    
    def secure_aggregation(self, 
                         updates: List[np.ndarray],
                         participant_masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Perform secure aggregation using masks.
        
        Allows aggregation without revealing individual contributions.
        """
        if len(updates) < self.aggregation_threshold:
            raise ValueError(f"Need at least {self.aggregation_threshold} participants")
        
        if participant_masks is None:
            # Generate random masks that sum to zero
            participant_masks = self._generate_zero_sum_masks(len(updates), updates[0].shape)
        
        # Add masks to updates
        masked_updates = [
            update + mask for update, mask in zip(updates, participant_masks)
        ]
        
        # Aggregate (masks cancel out)
        aggregated = np.sum(masked_updates, axis=0) / len(updates)
        
        return aggregated
    
    def _generate_zero_sum_masks(self, n_participants: int, shape: tuple) -> List[np.ndarray]:
        """Generate random masks that sum to zero."""
        masks = []
        
        # Generate n-1 random masks
        for i in range(n_participants - 1):
            mask = np.random.randn(*shape) * self.privacy_params["noise_scale"]
            masks.append(mask)
        
        # Last mask ensures sum is zero
        last_mask = -np.sum(masks, axis=0)
        masks.append(last_mask)
        
        return masks
    
    def privacy_preserving_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute similarity without revealing exact vectors.
        
        Uses randomized response for binary vectors.
        """
        # Add noise to vectors
        vec1_private = self.add_differential_privacy(vec1)
        vec2_private = self.add_differential_privacy(vec2)
        
        # Compute noisy similarity
        similarity = np.dot(vec1_private, vec2_private) / (
            np.linalg.norm(vec1_private) * np.linalg.norm(vec2_private)
        )
        
        return float(similarity)
    
    def homomorphic_aggregate(self, encrypted_values: List[Any]) -> Any:
        """
        Placeholder for homomorphic encryption aggregation.
        
        In practice, would use libraries like SEAL or TenSEAL.
        """
        # Simplified: just return sum for demonstration
        # Real implementation would maintain encryption
        return sum(encrypted_values)


class ByzantineFaultDetector:
    """
    Detects and handles Byzantine (malicious) participants in distributed COEC.
    """
    
    def __init__(self, tolerance_fraction: float = 0.3):
        self.tolerance_fraction = tolerance_fraction
        self.participant_history = defaultdict(list)
        self.trust_scores = defaultdict(lambda: 1.0)
    
    def validate_update(self, update: ParticipantUpdate, 
                       expected_bounds: Dict[str, Tuple[float, float]]) -> bool:
        """
        Validate that update is within expected bounds.
        
        Returns True if update appears legitimate.
        """
        # Check substrate delta magnitude
        delta_norm = np.linalg.norm(update.substrate_delta)
        if delta_norm > expected_bounds.get('delta_norm', (0, 10))[1]:
            return False
        
        # Check energy is reasonable
        if not expected_bounds.get('energy', (0, 1e6))[0] <= update.local_energy <= expected_bounds.get('energy', (0, 1e6))[1]:
            return False
        
        # Check constraint satisfactions are in [0, 1]
        for satisfaction in update.constraint_satisfactions.values():
            if not 0 <= satisfaction <= 1:
                return False
        
        return True
    
    def detect_outliers(self, updates: List[ParticipantUpdate]) -> List[str]:
        """
        Detect outlier updates that may be Byzantine.
        
        Returns list of suspicious participant IDs.
        """
        if len(updates) < 4:
            return []  # Need enough participants for statistical detection
        
        # Extract features from updates
        features = []
        for update in updates:
            feature = [
                np.linalg.norm(update.substrate_delta),
                update.local_energy,
                np.mean(list(update.constraint_satisfactions.values()))
            ]
            features.append(feature)
        
        features = np.array(features)
        
        # Compute median and MAD (Median Absolute Deviation)
        median = np.median(features, axis=0)
        mad = np.median(np.abs(features - median), axis=0)
        
        # Identify outliers (modified z-score > 3.5)
        outliers = []
        for i, (update, feature) in enumerate(zip(updates, features)):
            modified_z_score = 0.6745 * (feature - median) / (mad + 1e-10)
            if np.any(np.abs(modified_z_score) > 3.5):
                outliers.append(update.participant_id)
        
        return outliers
    
    def update_trust_scores(self, updates: List[ParticipantUpdate], outliers: List[str]):
        """Update trust scores based on behavior."""
        for update in updates:
            if update.participant_id in outliers:
                # Decrease trust
                self.trust_scores[update.participant_id] *= 0.9
            else:
                # Increase trust slowly
                self.trust_scores[update.participant_id] = min(
                    1.0, self.trust_scores[update.participant_id] * 1.01
                )
    
    def compute_byzantine_robust_aggregate(self, 
                                         updates: List[ParticipantUpdate]) -> np.ndarray:
        """
        Compute aggregate that's robust to Byzantine participants.
        
        Uses geometric median or trimmed mean.
        """
        # Filter out untrusted participants
        trusted_updates = [
            u for u in updates 
            if self.trust_scores[u.participant_id] > 0.5
        ]
        
        if len(trusted_updates) < len(updates) * (1 - self.tolerance_fraction):
            raise ValueError("Too many untrusted participants")
        
        # Extract substrate deltas
        deltas = [u.substrate_delta for u in trusted_updates]
        
        # Compute geometric median (robust to outliers)
        return self._geometric_median(deltas)
    
    def _geometric_median(self, points: List[np.ndarray], 
                         max_iter: int = 100, eps: float = 1e-5) -> np.ndarray:
        """
        Compute geometric median of points.
        
        More robust than mean to outliers.
        """
        points = np.array(points)
        
        # Initialize at centroid
        median = np.mean(points, axis=0)
        
        for _ in range(max_iter):
            # Compute distances to current median
            distances = np.array([
                np.linalg.norm(p - median) for p in points
            ])
            
            # Avoid division by zero
            distances = np.maximum(distances, eps)
            
            # Weighted average (Weiszfeld algorithm)
            weights = 1.0 / distances
            weights = weights / np.sum(weights)
            
            new_median = np.sum(points * weights[:, np.newaxis], axis=0)
            
            # Check convergence
            if np.linalg.norm(new_median - median) < eps:
                break
            
            median = new_median
        
        return median


class FederatedCOECSystem:
    """
    Federated learning system for distributed COEC.
    
    Allows multiple participants to collaboratively solve COEC problems
    while preserving privacy and handling Byzantine participants.
    """
    
    def __init__(self,
                 base_system: COECSystem,
                 num_participants: int,
                 privacy_level: str = "moderate",
                 byzantine_tolerance: float = 0.3):
        
        self.base_system = base_system
        self.num_participants = num_participants
        
        # Privacy and security components
        self.privacy_transform = PrivacyPreservingTransform(privacy_level)
        self.byzantine_detector = ByzantineFaultDetector(byzantine_tolerance)
        
        # Participant systems (local copies)
        self.participant_systems = []
        for i in range(num_participants):
            # Create local copy with same structure
            local_substrate = Substrate(
                self.base_system.substrate.dimensions,
                self.base_system.substrate.size
            )
            
            local_system = COECSystem(
                local_substrate,
                self.base_system.constraints,
                self.base_system.energy_landscape,
                self.base_system.evolver
            )
            
            self.participant_systems.append({
                'id': f'participant_{i}',
                'system': local_system,
                'data_distribution': self._generate_data_distribution(i)
            })
    
    def _generate_data_distribution(self, participant_idx: int) -> Callable:
        """
        Generate data distribution for participant.
        
        Simulates non-IID data across participants.
        """
        # Simple example: different regions of state space
        def distribution(base_state: np.ndarray) -> np.ndarray:
            # Add participant-specific bias
            bias = np.sin(participant_idx * np.pi / self.num_participants)
            noise = np.random.randn(*base_state.shape) * 0.1
            
            return base_state + bias + noise
        
        return distribution
    
    def federated_round(self, global_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one round of federated learning.
        
        Returns updated global state and metadata.
        """
        updates = []
        
        # Each participant computes local update
        for participant in self.participant_systems:
            # Apply local data distribution
            local_state = participant['data_distribution'](global_state)
            participant['system'].substrate.set_state(local_state)
            
            # Local evolution
            local_result = participant['system'].evolve(steps=10)
            
            # Compute update
            update = ParticipantUpdate(
                participant_id=participant['id'],
                substrate_delta=local_result.final_state - local_state,
                constraint_satisfactions=local_result.constraint_satisfaction,
                local_energy=local_result.final_energy,
                metadata={'steps': 10}
            )
            
            updates.append(update)
        
        # Add privacy noise
        private_updates = []
        for update in updates:
            private_delta = self.privacy_transform.add_differential_privacy(
                update.substrate_delta
            )
            private_update = ParticipantUpdate(
                participant_id=update.participant_id,
                substrate_delta=private_delta,
                constraint_satisfactions=update.constraint_satisfactions,
                local_energy=update.local_energy,
                metadata=update.metadata
            )
            private_updates.append(private_update)
        
        # Detect Byzantine participants
        outliers = self.byzantine_detector.detect_outliers(private_updates)
        self.byzantine_detector.update_trust_scores(private_updates, outliers)
        
        # Compute robust aggregate
        aggregate_delta = self.byzantine_detector.compute_byzantine_robust_aggregate(
            private_updates
        )
        
        # Update global state
        new_global_state = global_state + aggregate_delta
        
        metadata = {
            'num_updates': len(updates),
            'byzantine_detected': outliers,
            'trust_scores': dict(self.byzantine_detector.trust_scores),
            'privacy_level': self.privacy_transform.privacy_level
        }
        
        return new_global_state, metadata
    
    def run_federated_evolution(self, 
                               initial_state: np.ndarray,
                               num_rounds: int = 100) -> COECResult:
        """
        Run complete federated COEC evolution.
        
        Returns final result after all rounds.
        """
        trajectory = [initial_state]
        current_state = initial_state
        
        all_metadata = []
        
        for round_idx in range(num_rounds):
            # Execute federated round
            new_state, round_metadata = self.federated_round(current_state)
            
            trajectory.append(new_state)
            current_state = new_state
            
            round_metadata['round'] = round_idx
            all_metadata.append(round_metadata)
            
            # Check convergence
            if round_idx > 0:
                change = np.linalg.norm(trajectory[-1] - trajectory[-2])
                if change < 1e-6:
                    print(f"Converged at round {round_idx}")
                    break
        
        # Compute final metrics on global system
        self.base_system.substrate.set_state(current_state)
        final_energy = self.base_system.energy_landscape.compute_energy(current_state)
        
        final_satisfactions = {
            c.name: c.satisfaction(current_state)
            for c in self.base_system.constraints
        }
        
        return COECResult(
            trajectory=np.array(trajectory),
            final_state=current_state,
            final_energy=final_energy,
            constraint_satisfaction=final_satisfactions,
            metadata={'federated_rounds': all_metadata}
        )


class DistributedCOECCoordinator:
    """
    Coordinator for fully distributed COEC computation.
    
    Implements random task assignment and committee validation for
    decentralized execution.
    """
    
    def __init__(self,
                 num_nodes: int,
                 committee_size: int = 5,
                 consensus_threshold: float = 0.67):
        
        self.num_nodes = num_nodes
        self.committee_size = committee_size
        self.consensus_threshold = consensus_threshold
        
        # Node registry
        self.nodes = {
            f'node_{i}': {
                'id': f'node_{i}',
                'reputation': 1.0,
                'tasks_completed': 0
            }
            for i in range(num_nodes)
        }
    
    def assign_task_randomly(self, task_id: str) -> List[str]:
        """
        Randomly assign task to committee of nodes.
        
        Prevents collusion by random selection.
        """
        available_nodes = list(self.nodes.keys())
        
        # Weight by reputation
        weights = [self.nodes[node]['reputation'] for node in available_nodes]
        weights = np.array(weights) / np.sum(weights)
        
        # Random selection without replacement
        committee = np.random.choice(
            available_nodes,
            size=min(self.committee_size, len(available_nodes)),
            replace=False,
            p=weights
        )
        
        return list(committee)
    
    def validate_computation(self,
                           results: Dict[str, Any],
                           committee: List[str]) -> Tuple[bool, Any]:
        """
        Validate computation results from committee.
        
        Returns (is_valid, consensus_result).
        """
        if len(results) < self.committee_size * self.consensus_threshold:
            return False, None
        
        # Group similar results
        result_groups = defaultdict(list)
        
        for node_id, result in results.items():
            # Hash result for grouping
            result_hash = self._hash_result(result)
            result_groups[result_hash].append((node_id, result))
        
        # Find majority group
        majority_group = max(result_groups.values(), key=len)
        
        if len(majority_group) >= self.committee_size * self.consensus_threshold:
            # Update reputations
            all_nodes = set(committee)
            majority_nodes = {node_id for node_id, _ in majority_group}
            
            for node_id in majority_nodes:
                self.nodes[node_id]['reputation'] *= 1.01
                self.nodes[node_id]['tasks_completed'] += 1
            
            for node_id in all_nodes - majority_nodes:
                self.nodes[node_id]['reputation'] *= 0.95
            
            # Return consensus result
            return True, majority_group[0][1]
        
        return False, None
    
    def _hash_result(self, result: Any) -> str:
        """Create hash of computation result for comparison."""
        if isinstance(result, np.ndarray):
            # Round to avoid floating point differences
            rounded = np.round(result, decimals=6)
            result_str = rounded.tobytes()
        else:
            result_str = str(result).encode()
        
        return hashlib.sha256(result_str).hexdigest()
    
    def distribute_coec_computation(self,
                                   coec_system: COECSystem,
                                   num_tasks: int = 10) -> List[COECResult]:
        """
        Distribute COEC computation across network.
        
        Splits evolution into tasks and distributes to committees.
        """
        results = []
        
        for task_idx in range(num_tasks):
            task_id = f"task_{task_idx}"
            
            # Assign to committee
            committee = self.assign_task_randomly(task_id)
            
            # Simulate committee members computing
            # (In practice, would be actual distributed execution)
            committee_results = {}
            
            for node_id in committee:
                # Add some randomness to simulate different nodes
                if np.random.random() > 0.1:  # 90% honest
                    result = coec_system.evolve(steps=100)
                    committee_results[node_id] = result.final_state
                else:
                    # Byzantine node returns garbage
                    committee_results[node_id] = np.random.randn(
                        *coec_system.substrate.state.shape
                    )
            
            # Validate results
            is_valid, consensus_result = self.validate_computation(
                committee_results, committee
            )
            
            if is_valid:
                # Create result object
                final_energy = coec_system.energy_landscape.compute_energy(
                    consensus_result
                )
                
                result = COECResult(
                    trajectory=np.array([consensus_result]),
                    final_state=consensus_result,
                    final_energy=final_energy,
                    constraint_satisfaction={},
                    metadata={'committee': committee, 'task_id': task_id}
                )
                
                results.append(result)
        
        return results
