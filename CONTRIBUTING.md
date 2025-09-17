# Contributing to Market Sentiment Analysis for Investors

Thank you for considering contributing to this project! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code contributions.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

1. **Fork** the repository on GitHub
2. **Clone** the project to your own machine
3. **Commit** changes to your own branch
4. **Push** your work back up to your fork
5. Submit a **Pull Request** so that we can review your changes

## Development Setup

1. Set up your development environment:
   ```bash
   # Clone the repository
   git clone https://github.com/your-username/Market-Sentiment-Analysis-for-Investors-.git
   cd Market-Sentiment-Analysis-for-Investors
   
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
4. The PR must pass all CI checks before it can be merged.
5. The PR must be reviewed by at least one maintainer before merging.

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code
- Use type hints for all function signatures
- Write docstrings for all public functions and classes following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep functions small and focused on a single responsibility
- Write meaningful commit messages following [Conventional Commits](https://www.conventionalcommits.org/)

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a PR
- Update tests when fixing bugs or adding features
- Test coverage should not decrease

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src tests/
```

## Documentation

- Update documentation for any changes to the codebase
- Add docstrings to all new functions and classes
- Update README.md for any changes to setup or usage
- Add comments to explain complex logic

## Reporting Issues

When reporting issues, please include:
- Description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Version of the software

## Feature Requests

We welcome feature requests! Please open an issue with:
- A clear description of the feature
- The motivation for the feature
- Any alternative solutions or features you've considered
- Additional context or screenshots about the feature request

## Code Review Process

1. A maintainer will review your PR
2. You may receive feedback or be asked to make changes
3. Once approved, your PR will be merged into the main branch
4. The changes will be included in the next release

Thank you for contributing! Your help makes this project better for everyone. 🎉
