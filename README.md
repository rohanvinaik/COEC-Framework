# COEC Framework

Implementation of Constraint-Oriented Emergent Computation (COEC) - a substrate-independent framework for understanding computation as the trajectory of physical or biological systems through constrained state spaces.

## Overview

COEC reconceptualizes computation not as discrete logic or symbolic manipulation, but as systems undergoing entropy-driven transitions within boundary conditions. This framework provides a unified mathematical language for understanding computational processes across diverse biological contexts—from protein folding to neural dynamics.

## Key Concepts

- **Computation as Constraint Satisfaction**: Systems compute by navigating constrained state spaces
- **Distributed Agency**: No central controller; behavior emerges from interacting constraints
- **Substrate Independence**: Applies equally to molecular, cellular, tissue, and ecosystem processes
- **Information-Theoretic Foundation**: Integrates thermodynamic and informational principles

## Installation

```bash
git clone https://github.com/rohanvinaik/COEC-Framework.git
cd COEC-Framework
pip install -e .
```

## Quick Start

```python
from coec import Substrate, Constraint, COECSystem
from coec.constraints import EnergeticConstraint, TopologicalConstraint
from coec.evolution import GradientDescentEvolver

# Create a simple protein folding simulation
substrate = Substrate(dimensions=3, size=50)  # 50 amino acid chain

# Define constraints
energy_constraint = EnergeticConstraint(
    potential="lennard_jones",
    precision=0.8
)

topology_constraint = TopologicalConstraint(
    connectivity="chain",
    precision=1.0
)

# Create and run system
system = COECSystem(
    substrate=substrate,
    constraints=[energy_constraint, topology_constraint],
    evolver=GradientDescentEvolver(learning_rate=0.01)
)

result = system.evolve(steps=1000)
print(f"Final energy: {result.final_energy}")
```

## COEC Classes

The framework implements several classes of COEC systems:

- **SS-COEC** (Static-Structural): Systems that compute by reaching stable structures
- **DB-COEC** (Dynamic-Behavioral): Systems that produce stable temporal patterns
- **DM-COEC** (Distributed-Multiplicative): Computation emerges from multiple interacting subsystems
- **AP-COEC** (Adaptive-Plastic): Systems that modify their own constraints over time
- **PP-COEC** (Predictive-Probabilistic): Systems using internal models to anticipate future states

## Documentation

See the [docs](docs/) folder for detailed documentation and theoretical background.

## Examples

Check out the [examples](examples/) directory for:
- Protein folding simulations
- Neural dynamics modeling
- Swarm behavior demonstrations
- Step-by-step tutorials

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this framework in your research, please cite:

```
@software{coec_framework,
  title = {COEC Framework: Constraint-Oriented Emergent Computation},
  author = {Vinaik, Rohan},
  year = {2024},
  url = {https://github.com/rohanvinaik/COEC-Framework}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
