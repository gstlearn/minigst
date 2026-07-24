# Contributing to minigst Python Package

Thank you for your interest in contributing to the minigst Python package!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Install in development mode**:
   ```bash
   cd python
   pip install -e ".[dev]"
   ```

## Development Setup

### Prerequisites
- Python 3.8 or higher
- gstlearn Python package
- Development dependencies: pytest, black, flake8

### Install Development Dependencies
```bash
pip install pytest black flake8 jupyter
```

## Code Style

### Python Conventions
- Follow PEP 8 style guide
- Use snake_case for function and variable names
- Use descriptive variable names
- Add docstrings to all public functions

### Docstring Format
```python
def function_name(param1, param2):
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Examples:
        >>> import minigst as mg
        >>> result = mg.function_name(arg1, arg2)
    """
    # Implementation
```

## Testing

### Running Tests
```bash
cd python
pytest tests/
```

### Writing Tests
- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Include both positive and negative test cases

Example test:
```python
def test_df_to_db():
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "z": [1, 2]})
    db = mg.df_to_db(df, coord_names=["x", "y"])
    assert db.NSample() == 2
```

## Adding New Features

1. **Discuss first**: Open an issue to discuss the feature
2. **Write tests**: Add tests for the new functionality
3. **Implement**: Write the code following style guidelines
4. **Document**: Add docstrings and update documentation
5. **Examples**: Add usage examples if appropriate

## Documentation

### Updating Documentation
When adding or modifying functions:
1. Update docstrings in the code
2. Update `API_REFERENCE.md` if needed
3. Add examples to `README.md` if appropriate
4. Update `R_TO_PYTHON_MAPPING.md` if adding R equivalents

### Building Documentation
```bash
# Generate API documentation
python -m pydoc minigst > docs/api.html
```

## Pull Request Process

1. **Update your fork** with the latest main branch
2. **Create a feature branch** from main
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the guidelines above
4. **Test your changes** thoroughly
5. **Commit your changes** with clear commit messages
   ```bash
   git commit -m "Add feature: brief description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** on GitHub

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files included

## Reporting Bugs

When reporting bugs, please include:
1. **Python version** (`python --version`)
2. **minigst version**
3. **gstlearn version**
4. **Minimal reproducible example**
5. **Expected behavior**
6. **Actual behavior**
7. **Error messages** (full traceback)

## Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
1. **Clear description** of the enhancement
2. **Use case**: Why would this be useful?
3. **Example usage**: How would it work?
4. **Comparison**: How does the R version handle this?

## Code Review Process

- Maintainers will review pull requests
- Feedback may be provided for improvements
- Once approved, changes will be merged

## Questions?

If you have questions about contributing:
1. Check existing issues and pull requests
2. Open a new issue with your question
3. Tag it as "question"

## License

By contributing, you agree that your contributions will be licensed under the BSD-3 license.

Thank you for contributing to minigst!
