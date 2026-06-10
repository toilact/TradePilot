```markdown
# TradePilot Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the TradePilot Python codebase. You'll learn how to structure code, write and name files, manage imports and exports, follow commit message conventions, and understand the testing approach. This guide ensures consistency and maintainability in your contributions to TradePilot.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python files.
  - Example: `trade_engine.py`, `order_manager.py`

### Import Style
- Use **alias imports** for external and internal modules.
  - Example:
    ```python
    import numpy as np
    import trade_utils as tu
    ```

### Export Style
- Use **default exports** (i.e., define main classes or functions at the module level).
  - Example:
    ```python
    # In trade_engine.py
    class TradeEngine:
        pass
    ```

### Commit Messages
- Follow **conventional commit** format.
- Use prefixes like `feat` for features and `docs` for documentation.
- Keep commit messages concise (average ~70 characters).
  - Example:
    ```
    feat: add order matching logic to trade engine
    docs: update README with setup instructions
    ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new functionality.
**Command:** `/add-feature`

1. Create a new Python file using snake_case if needed.
2. Write your code, using alias imports and default exports.
3. Add or update tests in a corresponding `*.test.*` file.
4. Commit your changes with a `feat:` prefix.
    ```
    feat: implement trade validation logic
    ```
5. Push your changes and open a pull request.

### Updating Documentation
**Trigger:** When improving or adding documentation.
**Command:** `/update-docs`

1. Edit or create documentation files as needed.
2. Commit your changes with a `docs:` prefix.
    ```
    docs: add usage examples to API docs
    ```
3. Push your changes and open a pull request.

### Running Tests
**Trigger:** Before merging or after making changes.
**Command:** `/run-tests`

1. Locate all test files matching `*.test.*`.
2. Run tests using your preferred Python test runner (e.g., pytest, unittest).
    ```bash
    pytest
    ```
3. Ensure all tests pass before merging.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `trade_engine.test.py`).
- The specific testing framework is not enforced, but common Python test runners like `pytest` or `unittest` can be used.
- Place tests alongside or in a dedicated tests directory, as appropriate.

**Example test file:**
```python
# trade_engine.test.py
import unittest
from trade_engine import TradeEngine

class TestTradeEngine(unittest.TestCase):
    def test_order_execution(self):
        engine = TradeEngine()
        # Add assertions here
```

## Commands
| Command        | Purpose                                  |
|----------------|------------------------------------------|
| /add-feature   | Start workflow for adding a new feature  |
| /update-docs   | Start workflow for updating documentation|
| /run-tests     | Run all tests in the codebase            |
```
