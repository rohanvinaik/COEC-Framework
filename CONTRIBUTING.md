# Contributing to COEC Framework

Thank you for your interest in contributing to the Constraint-Oriented Emergent Computation (COEC) Framework! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [Issues](https://github.com/rohanvinaik/COEC-Framework/issues) section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (Python version, OS, etc.)

### Suggesting Enhancements

1. Open an issue with the "enhancement" label
2. Describe the feature and its motivation
3. Provide examples of how it would be used

### Code Contributions

#### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/COEC-Framework.git
cd COEC-Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

#### Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards

3. Add tests for new functionality

4. Run tests and linting:
   ```bash
   # Run tests
   pytest

   # Run linting
   flake8 coec tests
   mypy coec

   # Format code
   black coec tests
   isort coec tests
   ```

5. Commit your changes with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

6. Push to your fork and create a pull request

### Coding Standards

#### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names

#### Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings:

```python
def compute_energy(substrate: Substrate, parameters: Dict[str, float]) -> float:
    """
    Compute the energy of a substrate configuration.
    
    Args:
        substrate: The substrate instance to evaluate
        parameters: Dictionary of energy function parameters
        
    Returns:
        The computed energy value
        
    Raises:
        ValueError: If parameters are invalid
    """
```

#### Testing

- Write tests for all new functionality
- Place tests in `tests/` mirroring the package structure
- Use pytest for testing
- Aim for >80% code coverage

Example test:

```python
def test_linear_constraint_satisfaction():
    """Test that linear constraint correctly evaluates satisfaction."""
    substrate = EuclideanSubstrate(state=np.array([1.0, 2.0, 3.0]))
    constraint = LinearConstraint(w=np.array([1, 0, 0]), b=0.5)
    
    satisfaction = constraint.evaluate(substrate)
    assert 0 <= satisfaction <= 1
    assert satisfaction > 0.5  # Should be satisfied since 1.0 > 0.5
```

### Areas for Contribution

#### Core Framework

- New substrate types (e.g., tensor networks, manifolds)
- Additional constraint classes (e.g., information-theoretic, quantum)
- Energy landscape implementations
- Evolution operators (e.g., genetic algorithms, simulated annealing)

#### Applications

- Domain-specific examples (biology, physics, engineering)
- Integration with existing tools (PyTorch, JAX, etc.)
- Visualization utilities
- Performance optimizations

#### Documentation

- Tutorials and guides
- Mathematical derivations
- API documentation improvements
- Translation to other languages

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

### Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- PATCH: Bug fixes, documentation updates
- MINOR: New features, backward compatible
- MAJOR: Breaking changes

## Questions?

Feel free to open an issue for any questions about contributing!

## Recognition

Contributors will be acknowledged in:
- The project README
- Release notes
- Academic publications using the framework

Thank you for helping make COEC better!