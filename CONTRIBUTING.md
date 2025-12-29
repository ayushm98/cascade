# Contributing to Cascade

Thank you for your interest in contributing to Cascade! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/cascade.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `make test`
6. Commit your changes: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Install dependencies
pip install poetry
poetry install

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Start development server
make run
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for public functions
- Keep functions focused and small
- Add tests for new features

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Test both happy path and edge cases

## Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example: `feat: add semantic caching with Qdrant`

## Pull Request Process

1. Update README.md with details of changes if needed
2. Update documentation
3. Add tests for new functionality
4. Ensure CI/CD pipeline passes
5. Request review from maintainers

## Questions?

Feel free to open an issue for questions or discussions!
