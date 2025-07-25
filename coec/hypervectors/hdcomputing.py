"""
Hyperdimensional Computing Operations for COEC.

This module implements the hypervector operations that enable semantic
embedding and associative memory in COEC systems.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from scipy.spatial.distance import cosine


class Hypervector:
    """
    High-dimensional vector representation for COEC semantic embedding.
    
    Implements binding, bundling, and similarity operations as described
    in the COEC framework (Section 2.9).
    """
    
    def __init__(self, dimension: int = 10000, data: Optional[np.ndarray] = None):
        self.dimension = dimension
        
        if data is not None:
            assert len(data) == dimension, f"Data must have dimension {dimension}"
            self.data = data.astype(np.float32)
        else:
            # Initialize with random Gaussian values
            self.data = np.random.randn(dimension).astype(np.float32)
            self.normalize()
    
    def normalize(self):
        """Normalize the hypervector to unit length."""
        norm = np.linalg.norm(self.data)
        if norm > 0:
            self.data = self.data / norm
    
    def bind(self, other: 'Hypervector') -> 'Hypervector':
        """
        Binding operation (⊗) - element-wise multiplication.
        
        Used to create associations between concepts.
        """
        assert self.dimension == other.dimension, "Dimensions must match"
        result_data = self.data * other.data
        return Hypervector(self.dimension, result_data)
    
    def bundle(self, other: 'Hypervector') -> 'Hypervector':
        """
        Bundling operation (⊕) - normalized addition.
        
        Used to create superposition of multiple concepts.
        """
        assert self.dimension == other.dimension, "Dimensions must match"
        result_data = self.data + other.data
        result = Hypervector(self.dimension, result_data)
        result.normalize()
        return result
    
    def permute(self, shifts: int = 1) -> 'Hypervector':
        """
        Permutation operation - circular shift.
        
        Used for representing sequences and positions.
        """
        result_data = np.roll(self.data, shifts)
        return Hypervector(self.dimension, result_data)
    
    def similarity(self, other: 'Hypervector') -> float:
        """
        Compute cosine similarity between hypervectors.
        
        Returns value in [-1, 1], where 1 means identical.
        """
        assert self.dimension == other.dimension, "Dimensions must match"
        return 1 - cosine(self.data, other.data)
    
    def __mul__(self, other: 'Hypervector') -> 'Hypervector':
        """Operator overload for binding."""
        return self.bind(other)
    
    def __add__(self, other: 'Hypervector') -> 'Hypervector':
        """Operator overload for bundling."""
        return self.bundle(other)


class HypervectorSpace:
    """
    Manager for hypervector operations and semantic embeddings.
    
    Implements the hypervector semantic embedding described in
    COEC Section 2.9 and Definition 7.
    """
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.item_memory = {}  # Store named hypervectors
        
        # Create base hypervectors for common concepts
        self._initialize_base_vectors()
    
    def _initialize_base_vectors(self):
        """Create orthogonal base vectors for fundamental concepts."""
        # Position encoding vectors
        self.item_memory['POS_START'] = Hypervector(self.dimension)
        self.item_memory['POS_END'] = Hypervector(self.dimension)
        
        # Type encoding vectors
        self.item_memory['TYPE_AMINO'] = Hypervector(self.dimension)
        self.item_memory['TYPE_HYDROPHOBIC'] = Hypervector(self.dimension)
        self.item_memory['TYPE_HYDROPHILIC'] = Hypervector(self.dimension)
        
        # State encoding vectors
        self.item_memory['STATE_FOLDED'] = Hypervector(self.dimension)
        self.item_memory['STATE_UNFOLDED'] = Hypervector(self.dimension)
    
    def create_item(self, name: str) -> Hypervector:
        """Create and store a new named hypervector."""
        hv = Hypervector(self.dimension)
        self.item_memory[name] = hv
        return hv
    
    def get_item(self, name: str) -> Hypervector:
        """Retrieve a named hypervector."""
        if name not in self.item_memory:
            return self.create_item(name)
        return self.item_memory[name]
    
    def encode_sequence(self, items: List[str]) -> Hypervector:
        """
        Encode a sequence using position-sensitive bundling.
        
        Example: encode_sequence(['A', 'B', 'C']) creates a hypervector
        representing the ordered sequence A-B-C.
        """
        result = None
        
        for i, item_name in enumerate(items):
            item_hv = self.get_item(item_name)
            # Use permutation to encode position
            positioned_hv = item_hv.permute(i)
            
            if result is None:
                result = positioned_hv
            else:
                result = result + positioned_hv
        
        return result
    
    def encode_structure(self, structure_dict: dict) -> Hypervector:
        """
        Encode a complex structure with attributes.
        
        Example: encode_structure({
            'type': 'protein',
            'length': 50,
            'hydrophobic_positions': [5, 10, 15]
        })
        """
        result = None
        
        for key, value in structure_dict.items():
            key_hv = self.get_item(f"KEY_{key}")
            
            if isinstance(value, list):
                value_hv = self.encode_sequence([str(v) for v in value])
            else:
                value_hv = self.get_item(f"VAL_{value}")
            
            # Bind key and value
            pair_hv = key_hv * value_hv
            
            if result is None:
                result = pair_hv
            else:
                result = result + pair_hv
        
        return result
    
    def query_similarity(self, query_hv: Hypervector, 
                        threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Find stored items similar to query hypervector.
        
        Returns list of (name, similarity) tuples above threshold.
        """
        results = []
        
        for name, stored_hv in self.item_memory.items():
            sim = query_hv.similarity(stored_hv)
            if sim > threshold:
                results.append((name, sim))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class HypervectorHashLattice:
    """
    Efficient storage and retrieval structure for hypervectors.
    
    Implements the Hypervector-Hash Lattice described in Definition 7.1,
    providing O(log n) retrieval time for nearest-neighbor queries.
    """
    
    def __init__(self, dimension: int = 10000, hash_bits: int = 64):
        self.dimension = dimension
        self.hash_bits = hash_bits
        self.buckets = {}
        
        # Random projection matrix for LSH
        self.projection_matrix = np.random.randn(hash_bits, dimension)
        self.projection_matrix = self.projection_matrix.astype(np.float32)
    
    def _compute_hash(self, hypervector: Hypervector) -> str:
        """
        Compute locality-sensitive hash of hypervector.
        
        Uses random projection followed by binary quantization.
        """
        # Project to lower dimension
        projection = np.dot(self.projection_matrix, hypervector.data)
        
        # Binary quantization
        binary = (projection > 0).astype(int)
        
        # Convert to string hash
        hash_str = ''.join(map(str, binary))
        return hash_str
    
    def insert(self, name: str, hypervector: Hypervector):
        """Insert a named hypervector into the lattice."""
        hash_key = self._compute_hash(hypervector)
        
        if hash_key not in self.buckets:
            self.buckets[hash_key] = []
        
        self.buckets[hash_key].append((name, hypervector))
    
    def query_nearest(self, query_hv: Hypervector, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors to query hypervector.
        
        Uses LSH for efficient approximate search.
        """
        query_hash = self._compute_hash(query_hv)
        candidates = []
        
        # Check exact hash match first
        if query_hash in self.buckets:
            candidates.extend(self.buckets[query_hash])
        
        # Check nearby buckets (hamming distance 1)
        for i in range(self.hash_bits):
            # Flip one bit
            nearby_hash = list(query_hash)
            nearby_hash[i] = '1' if nearby_hash[i] == '0' else '0'
            nearby_hash = ''.join(nearby_hash)
            
            if nearby_hash in self.buckets:
                candidates.extend(self.buckets[nearby_hash])
        
        # Compute actual similarities
        results = []
        for name, hv in candidates:
            sim = query_hv.similarity(hv)
            results.append((name, sim))
        
        # Sort and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


def demonstrate_hypervector_operations():
    """
    Demonstrate basic hypervector operations for COEC.
    """
    print("=== Hypervector Operations Demo ===\n")
    
    # Create hypervector space
    hv_space = HypervectorSpace(dimension=1000)  # Smaller for demo
    
    # Example 1: Encode a protein sequence
    print("1. Encoding protein sequence:")
    sequence = ['MET', 'ALA', 'VAL', 'LEU', 'ILE']
    seq_hv = hv_space.encode_sequence(sequence)
    print(f"   Encoded sequence: {sequence}")
    print(f"   Hypervector norm: {np.linalg.norm(seq_hv.data):.3f}")
    
    # Example 2: Encode protein properties
    print("\n2. Encoding protein structure:")
    structure = {
        'length': 5,
        'hydrophobic_count': 3,
        'alpha_helix': True
    }
    struct_hv = hv_space.encode_structure(structure)
    print(f"   Structure: {structure}")
    
    # Example 3: Binding operation
    print("\n3. Binding sequence and structure:")
    protein_hv = seq_hv * struct_hv
    hv_space.item_memory['PROTEIN_1'] = protein_hv
    
    # Example 4: Similarity search
    print("\n4. Searching for similar items:")
    # Create a query that's similar but not identical
    query_seq = ['MET', 'ALA', 'VAL']
    query_hv = hv_space.encode_sequence(query_seq)
    
    similar_items = hv_space.query_similarity(query_hv, threshold=0.1)
    print(f"   Query sequence: {query_seq}")
    print("   Similar items found:")
    for name, sim in similar_items[:5]:
        print(f"     {name}: {sim:.3f}")
    
    # Example 5: Hash lattice for efficient retrieval
    print("\n5. Testing hash lattice efficiency:")
    lattice = HypervectorHashLattice(dimension=1000, hash_bits=32)
    
    # Insert many items
    for i in range(100):
        name = f"PROTEIN_{i}"
        hv = Hypervector(1000)
        lattice.insert(name, hv)
    
    # Query nearest neighbors
    nearest = lattice.query_nearest(protein_hv, k=3)
    print("   Nearest neighbors to PROTEIN_1:")
    for name, sim in nearest:
        print(f"     {name}: {sim:.3f}")


if __name__ == "__main__":
    demonstrate_hypervector_operations()
