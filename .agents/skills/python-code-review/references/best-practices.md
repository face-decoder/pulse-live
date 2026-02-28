# Python Best Practices Guide (2025-2026)

A comprehensive guide to writing clean, maintainable, and professional Python code.

---

## Table of Contents

1. [Code Style and Formatting (PEP 8)](#1-code-style-and-formatting-pep-8)
2. [Naming Conventions](#2-naming-conventions)
3. [Type Hints and Static Typing](#3-type-hints-and-static-typing)
4. [Project Structure](#4-project-structure)
5. [Dependency Management](#5-dependency-management)
6. [Error Handling](#6-error-handling)
7. [Logging](#7-logging)
8. [Testing](#8-testing)
9. [Documentation](#9-documentation)
10. [Async Programming](#10-async-programming)
11. [Performance Optimization](#11-performance-optimization)
12. [Security Best Practices](#12-security-best-practices)
13. [Modern Python Features](#13-modern-python-features)
14. [Code Quality Tools](#14-code-quality-tools)
15. [Design Patterns and Principles](#15-design-patterns-and-principles)

---

## 1. Code Style and Formatting (PEP 8)

PEP 8 is the official style guide for Python code. Following these conventions ensures consistency and readability.

### Indentation

Use 4 spaces per indentation level. Never mix tabs and spaces.

```python
# Good
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total

# Bad - using 2 spaces or tabs
def calculate_total(items):
  total = 0
  for item in items:
    total += item.price
  return total
```

### Line Length

- Limit lines to 79 characters for code
- Limit lines to 72 characters for docstrings and comments
- Use implicit line continuation inside parentheses, brackets, and braces

```python
# Good - implicit continuation
result = some_function(
    argument_one,
    argument_two,
    argument_three,
)

# Good - breaking long strings
message = (
    "This is a very long message that needs to be "
    "split across multiple lines for readability."
)

# Avoid - backslash continuation (use only when necessary)
result = some_very_long_variable_name + \
         another_very_long_variable_name
```

### Blank Lines

- Two blank lines around top-level function and class definitions
- One blank line around method definitions inside a class
- Use blank lines sparingly inside functions to indicate logical sections

```python
import os


class MyClass:
    """A sample class."""

    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1


def standalone_function():
    """A standalone function."""
    pass
```

### Imports

Imports should be on separate lines and grouped in the following order:
1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

```python
# Good
import os
import sys
from pathlib import Path

import httpx
import pydantic
from fastapi import FastAPI

from myproject.config import settings
from myproject.utils import helpers

# Bad - multiple imports on one line (except from imports)
import os, sys, json
```

### Whitespace

```python
# Good
spam(ham[1], {eggs: 2})
x = 1
y = 2
long_variable = 3

def complex(real, imag=0.0):
    return magic(r=real, i=imag)

# Bad
spam( ham[ 1 ], { eggs: 2 } )
x             = 1
y             = 2
long_variable = 3

def complex(real, imag = 0.0):
    return magic(r = real, i = imag)
```

### String Quotes

Pick either single or double quotes and be consistent. Use the other to avoid escaping.

```python
# Consistent - pick one style
name = "Alice"
message = "Hello, World!"

# Use other quote to avoid escaping
sql = "SELECT * FROM users WHERE name = 'Alice'"
html = '<div class="container">Content</div>'

# Triple double quotes for docstrings (PEP 257)
def my_function():
    """This is a docstring."""
    pass
```

---

## 2. Naming Conventions

### Variables and Functions

Use `snake_case` for variables, functions, and methods.

```python
# Good
user_name = "alice"
total_count = 42
is_valid = True

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

def get_user_by_id(user_id):
    pass

# Bad
userName = "alice"  # camelCase
TotalCount = 42     # PascalCase
IsValid = True      # PascalCase
```

### Classes

Use `PascalCase` (CapWords) for class names.

```python
# Good
class UserAccount:
    pass

class HTTPRequestHandler:
    pass

class XMLParser:
    pass

# Bad
class user_account:
    pass

class httpRequestHandler:
    pass
```

### Constants

Use `UPPER_SNAKE_CASE` for constants.

```python
# Good
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# Bad
maxConnections = 100
default_timeout = 30
```

### Private and Protected

Use leading underscores to indicate visibility.

```python
class MyClass:
    def __init__(self):
        self.public_attr = 1        # Public
        self._protected_attr = 2    # Protected (convention)
        self.__private_attr = 3     # Private (name mangling)

    def public_method(self):
        pass

    def _protected_method(self):
        """Internal use, but accessible."""
        pass

    def __private_method(self):
        """Name mangled to _MyClass__private_method."""
        pass
```

### Special Names to Avoid

```python
# Avoid these single-character names (look like numbers)
l = 1  # lowercase L looks like 1
O = 0  # uppercase O looks like 0
I = 1  # uppercase I looks like 1

# Better alternatives
length = 1
output = 0
index = 1
```

### Module and Package Names

Use short, all-lowercase names. Underscores can be used if it improves readability.

```python
# Good module names
mymodule.py
my_module.py
utils.py
helpers.py

# Good package names
mypackage/
my_package/
```

---

## 3. Type Hints and Static Typing

Type hints improve code readability, enable better IDE support, and catch bugs early.

### Basic Type Hints

```python
# Variables
name: str = "Alice"
age: int = 30
price: float = 19.99
is_active: bool = True

# Functions
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def process_data(data: bytes) -> None:
    # Function returns None
    print(data)
```

### Collection Types (Python 3.9+)

```python
# Use built-in generics (Python 3.9+)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 100, "Bob": 85}
coordinates: tuple[float, float] = (1.0, 2.0)
unique_ids: set[int] = {1, 2, 3}

# Variable-length tuples
values: tuple[int, ...] = (1, 2, 3, 4, 5)

# Nested collections
matrix: list[list[int]] = [[1, 2], [3, 4]]
user_data: dict[str, list[str]] = {"roles": ["admin", "user"]}
```

### Optional and Union Types (Python 3.10+)

```python
# Python 3.10+ syntax (preferred)
def find_user(user_id: int) -> User | None:
    """Returns User or None if not found."""
    pass

def process(value: int | str | float) -> str:
    """Accepts int, str, or float."""
    return str(value)

# For Python 3.9 and earlier
from typing import Optional, Union

def find_user(user_id: int) -> Optional[User]:
    pass

def process(value: Union[int, str, float]) -> str:
    return str(value)
```

### Type Aliases

```python
# Simple type aliases
UserId = int
Username = str
Coordinates = tuple[float, float]

# Complex type aliases
UserDict = dict[str, str | int | None]
Callback = Callable[[int, str], bool]
JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]

# Using TypeAlias (Python 3.10+)
from typing import TypeAlias

Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]
```

### Callable Types

```python
from collections.abc import Callable

# Function that takes int and str, returns bool
def register_callback(callback: Callable[[int, str], bool]) -> None:
    pass

# Function that takes any args and returns None
def log_call(func: Callable[..., None]) -> None:
    pass

# Async callable
from collections.abc import Awaitable

AsyncCallback = Callable[[str], Awaitable[int]]
```

### Generic Types

```python
from typing import TypeVar, Generic

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

# Constrained TypeVar
Number = TypeVar("Number", int, float)

def add_numbers(a: Number, b: Number) -> Number:
    return a + b

# Bound TypeVar
from typing import Comparable

Sortable = TypeVar("Sortable", bound=Comparable)
```

### Protocol (Structural Subtyping)

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

def render(shape: Drawable) -> None:
    shape.draw()

# Both Circle and Square are valid Drawable without explicit inheritance
render(Circle())
render(Square())
```

### TypedDict

```python
from typing import TypedDict, Required, NotRequired

class UserDict(TypedDict):
    name: str
    age: int
    email: str | None

# With optional fields (Python 3.11+)
class ConfigDict(TypedDict, total=False):
    debug: bool
    log_level: str

# Mixed required and optional
class RequestDict(TypedDict):
    method: Required[str]
    url: Required[str]
    headers: NotRequired[dict[str, str]]
    body: NotRequired[bytes]
```

### Self Type (Python 3.11+)

```python
from typing import Self

class Builder:
    def __init__(self) -> None:
        self.value = 0

    def set_value(self, value: int) -> Self:
        self.value = value
        return self

    def clone(self) -> Self:
        new = type(self)()
        new.value = self.value
        return new
```

### Best Practices for Type Hints

```python
# 1. Start with public API, then expand
# 2. Use abstract types for parameters, concrete for returns
from collections.abc import Sequence, Mapping, Iterable

def process_items(items: Sequence[int]) -> list[int]:  # Accept any sequence
    return [x * 2 for x in items]

# 3. Use | None instead of Optional (Python 3.10+)
def get_user(id: int) -> User | None:
    pass

# 4. Avoid Any when possible, but use it when truly needed
from typing import Any

def deserialize(data: bytes) -> Any:
    pass

# 5. Use TypeGuard for type narrowing
from typing import TypeGuard

def is_str_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

# 6. Use Final for constants
from typing import Final

MAX_SIZE: Final = 100
```

---

## 4. Project Structure

### Standard Python Package Structure

```
my_project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── __main__.py      # Entry point for python -m my_package
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   └── services.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py
│       │   └── middleware.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── helpers.py
│       └── config.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures
│   ├── test_core/
│   │   ├── __init__.py
│   │   └── test_models.py
│   └── test_api/
│       ├── __init__.py
│       └── test_routes.py
├── docs/
│   ├── conf.py              # Sphinx config
│   └── index.rst
├── scripts/
│   └── seed_data.py
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
└── LICENSE
```

### The `src` Layout

The `src` layout prevents accidentally importing from the local directory instead of the installed package.

```python
# With src layout, you MUST install the package to import it
# This catches import errors early

# pyproject.toml
[tool.setuptools.packages.find]
where = ["src"]
```

### `__init__.py` Best Practices

```python
# src/my_package/__init__.py

# Define public API
from my_package.core.models import User, Product
from my_package.core.services import UserService

# Version
__version__ = "1.0.0"

# Control what's exported with __all__
__all__ = [
    "User",
    "Product",
    "UserService",
    "__version__",
]
```

### `__main__.py` for CLI Entry Points

```python
# src/my_package/__main__.py
"""Entry point for python -m my_package."""

from my_package.cli import main

if __name__ == "__main__":
    main()
```

---

## 5. Dependency Management

### pyproject.toml (Modern Standard)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
version = "1.0.0"
description = "A sample Python package"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
keywords = ["sample", "package"]

dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
    "pre-commit>=3.0.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
]

[project.scripts]
my-cli = "my_package.cli:main"

[project.urls]
Homepage = "https://github.com/username/my-package"
Documentation = "https://my-package.readthedocs.io"
Repository = "https://github.com/username/my-package"

# Tool configurations
[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra -q --cov=my_package"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = []

[tool.ruff.lint.isort]
known-first-party = ["my_package"]
```

### Virtual Environments

```bash
# Create virtual environment
python -m venv .venv

# Activate (Unix/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install in editable mode for development
pip install -e ".[dev]"
```

### Using uv (Fast Package Manager)

```bash
# Install uv
pip install uv

# Create virtual environment
uv venv

# Install dependencies
uv pip install -e ".[dev]"

# Sync dependencies (like pip-tools)
uv pip compile pyproject.toml -o requirements.txt
uv pip sync requirements.txt
```

### Environment Variables

Use `python-dotenv` for local development:

```python
# .env file (never commit to git!)
DATABASE_URL=postgresql://localhost/mydb
SECRET_KEY=your-secret-key
DEBUG=true

# config.py
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
```

---

## 6. Error Handling

### Exception Hierarchy

```python
# Custom exception hierarchy
class AppError(Exception):
    """Base exception for application errors."""
    pass

class ValidationError(AppError):
    """Raised when validation fails."""
    pass

class NotFoundError(AppError):
    """Raised when a resource is not found."""
    pass

class AuthenticationError(AppError):
    """Raised when authentication fails."""
    pass

class AuthorizationError(AppError):
    """Raised when user lacks permissions."""
    pass
```

### Best Practices

```python
# 1. Catch specific exceptions
try:
    user = get_user(user_id)
except NotFoundError:
    return None
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    raise

# 2. Avoid bare except
# Bad
try:
    do_something()
except:  # Catches everything including KeyboardInterrupt
    pass

# Good
try:
    do_something()
except Exception as e:
    logger.exception("Unexpected error")
    raise

# 3. Keep try blocks small
# Bad
try:
    user = get_user(user_id)
    orders = get_orders(user.id)
    total = calculate_total(orders)
    send_email(user.email, total)
except Exception as e:
    # Which operation failed?
    pass

# Good
try:
    user = get_user(user_id)
except NotFoundError:
    return None

try:
    orders = get_orders(user.id)
except DatabaseError:
    logger.error("Failed to fetch orders")
    raise

# 4. Preserve exception chain
try:
    data = json.loads(raw_data)
except json.JSONDecodeError as e:
    raise ValidationError(f"Invalid JSON: {e}") from e

# 5. Use context managers for cleanup
with open("file.txt") as f:
    data = f.read()

# 6. Use finally for guaranteed cleanup
resource = acquire_resource()
try:
    process(resource)
finally:
    release_resource(resource)
```

### Exception Groups (Python 3.11+)

```python
# Handle multiple exceptions concurrently
async def fetch_all(urls: list[str]) -> list[Response]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(url)) for url in urls]
    return [t.result() for t in tasks]

# Catch exception groups
try:
    results = await fetch_all(urls)
except* ConnectionError as e:
    # Handle connection errors
    for exc in e.exceptions:
        logger.error(f"Connection failed: {exc}")
except* TimeoutError as e:
    # Handle timeouts
    pass
```

### EAFP vs LBYL

Python prefers EAFP (Easier to Ask for Forgiveness than Permission) over LBYL (Look Before You Leap).

```python
# LBYL (Less Pythonic)
if key in dictionary:
    value = dictionary[key]
else:
    value = default

# EAFP (More Pythonic)
try:
    value = dictionary[key]
except KeyError:
    value = default

# Even better - use dict methods
value = dictionary.get(key, default)

# LBYL (Less Pythonic)
if hasattr(obj, 'method'):
    obj.method()

# EAFP (More Pythonic)
try:
    obj.method()
except AttributeError:
    pass
```

---

## 7. Logging

### Basic Setup

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get logger for module
logger = logging.getLogger(__name__)

def process_data(data):
    logger.info("Processing data started")
    logger.debug(f"Data size: {len(data)}")

    try:
        result = transform(data)
        logger.info("Processing completed successfully")
        return result
    except Exception:
        logger.exception("Processing failed")  # Includes traceback
        raise
```

### Production Logging Configuration

```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
        "my_package": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Structured Logging with structlog

```python
import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Structured logging with context
logger.info(
    "user_login",
    user_id=123,
    ip_address="192.168.1.1",
    success=True,
)
```

### Logging Best Practices

```python
# 1. Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General operational information")
logger.warning("Warning about potential issues")
logger.error("Error that needs attention")
logger.critical("Critical error, application may crash")

# 2. Use lazy formatting (not f-strings for expensive operations)
# Bad - string always formatted
logger.debug(f"Processing {expensive_operation()}")

# Good - only formatted if debug is enabled
logger.debug("Processing %s", expensive_operation())

# 3. Include context
logger.info(
    "Order processed",
    extra={
        "order_id": order.id,
        "user_id": user.id,
        "total": order.total,
    }
)

# 4. Use exception() for errors with traceback
try:
    process()
except Exception:
    logger.exception("Processing failed")  # Automatically includes traceback

# 5. Don't log sensitive data
# Bad
logger.info(f"User login: {username}, password: {password}")

# Good
logger.info(f"User login: {username}")
```

---

## 8. Testing

### pytest Basics

```python
# tests/test_calculator.py
import pytest
from my_package.calculator import add, divide

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -1) == -2

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)

def test_divide_float_result():
    result = divide(5, 2)
    assert result == pytest.approx(2.5)
```

### Fixtures

```python
# tests/conftest.py
import pytest
from my_package.database import Database
from my_package.models import User

@pytest.fixture
def db():
    """Create a test database."""
    database = Database(":memory:")
    database.create_tables()
    yield database
    database.close()

@pytest.fixture
def sample_user(db):
    """Create a sample user."""
    user = User(name="Test User", email="test@example.com")
    db.save(user)
    return user

# Using fixtures
def test_user_creation(db, sample_user):
    retrieved = db.get_user(sample_user.id)
    assert retrieved.name == "Test User"
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("Python", "PYTHON"),
    ("", ""),
])
def test_uppercase(input, expected):
    assert input.upper() == expected

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

### Mocking

```python
from unittest.mock import Mock, patch, AsyncMock
import pytest

# Mocking a function
def test_send_email():
    with patch("my_package.email.smtp_send") as mock_send:
        mock_send.return_value = True
        result = send_welcome_email("user@example.com")
        assert result is True
        mock_send.assert_called_once_with(
            to="user@example.com",
            subject="Welcome!",
        )

# Mocking a class
def test_api_client():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "test"}

    with patch("my_package.api.httpx.get", return_value=mock_response):
        result = fetch_data()
        assert result == {"data": "test"}

# Async mocking
@pytest.mark.asyncio
async def test_async_fetch():
    with patch("my_package.api.aiohttp.get", new_callable=AsyncMock) as mock:
        mock.return_value.json = AsyncMock(return_value={"key": "value"})
        result = await async_fetch_data()
        assert result == {"key": "value"}
```

### Test Organization

```python
# Group related tests in classes
class TestUserService:
    """Tests for UserService."""

    def test_create_user(self, db):
        service = UserService(db)
        user = service.create("Alice", "alice@example.com")
        assert user.id is not None
        assert user.name == "Alice"

    def test_create_user_duplicate_email(self, db, sample_user):
        service = UserService(db)
        with pytest.raises(DuplicateEmailError):
            service.create("Bob", sample_user.email)

    def test_delete_user(self, db, sample_user):
        service = UserService(db)
        service.delete(sample_user.id)
        assert db.get_user(sample_user.id) is None
```

### Async Testing

```python
import pytest

@pytest.mark.asyncio
async def test_async_fetch():
    result = await fetch_data("https://api.example.com")
    assert result["status"] == "ok"

@pytest.mark.asyncio
async def test_async_context_manager():
    async with DatabaseConnection() as conn:
        result = await conn.execute("SELECT 1")
        assert result == 1
```

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["*/tests/*", "*/__main__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
fail_under = 80
```

---

## 9. Documentation

### Docstring Formats

```python
# Google Style (recommended)
def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetch data from a URL.

    Retrieves JSON data from the specified URL with configurable timeout.

    Args:
        url: The URL to fetch data from.
        timeout: Request timeout in seconds. Defaults to 30.

    Returns:
        A dictionary containing the response data.

    Raises:
        ConnectionError: If the connection fails.
        TimeoutError: If the request times out.

    Example:
        >>> data = fetch_data("https://api.example.com/data")
        >>> print(data["status"])
        'ok'
    """
    pass

# NumPy Style
def calculate_statistics(data: list[float]) -> tuple[float, float, float]:
    """
    Calculate basic statistics for a dataset.

    Parameters
    ----------
    data : list[float]
        A list of numerical values.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing (mean, median, std_dev).

    Examples
    --------
    >>> calculate_statistics([1, 2, 3, 4, 5])
    (3.0, 3.0, 1.4142135623730951)
    """
    pass
```

### Class Documentation

```python
class UserService:
    """Service for managing user operations.

    This service provides methods for creating, updating, and deleting
    users in the system. It handles validation and persistence.

    Attributes:
        db: Database connection instance.
        cache: Optional cache backend for performance.

    Example:
        >>> service = UserService(database)
        >>> user = service.create("Alice", "alice@example.com")
        >>> print(user.id)
        1
    """

    def __init__(self, db: Database, cache: Cache | None = None) -> None:
        """Initialize the UserService.

        Args:
            db: Database connection to use.
            cache: Optional cache backend. If not provided, caching is disabled.
        """
        self.db = db
        self.cache = cache
```

### Module Documentation

```python
"""User management module.

This module provides classes and functions for managing users in the
application. It includes user creation, authentication, and profile
management.

Typical usage example:

    from my_package.users import UserService, User

    service = UserService(database)
    user = service.create("Alice", "alice@example.com")

    if service.authenticate(user, "password"):
        print("Authenticated!")

Classes:
    User: Represents a user in the system.
    UserService: Service for user operations.

Functions:
    hash_password: Securely hash a password.
    verify_password: Verify a password against a hash.
"""
```

---

## 10. Async Programming

### asyncio Basics

```python
import asyncio

async def fetch_data(url: str) -> dict:
    """Fetch data from URL asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

async def main():
    # Run single coroutine
    data = await fetch_data("https://api.example.com/data")
    print(data)

# Entry point
asyncio.run(main())
```

### Concurrent Execution

```python
import asyncio

async def fetch_all(urls: list[str]) -> list[dict]:
    """Fetch multiple URLs concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# With error handling
async def fetch_all_safe(urls: list[str]) -> list[dict | None]:
    """Fetch URLs, returning None for failures."""
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r.json() if not isinstance(r, Exception) else None
            for r in results
        ]
```

### Task Groups (Python 3.11+)

```python
import asyncio

async def process_batch(items: list[str]) -> list[Result]:
    """Process items with automatic cancellation on error."""
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(process_item(item)) for item in items]
    return [task.result() for task in tasks]

# If any task fails, all others are cancelled automatically
```

### Async Context Managers

```python
class AsyncDatabaseConnection:
    """Async database connection with context manager."""

    async def __aenter__(self) -> "AsyncDatabaseConnection":
        self.conn = await asyncpg.connect(DATABASE_URL)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.conn.close()

    async def execute(self, query: str) -> list:
        return await self.conn.fetch(query)

# Usage
async with AsyncDatabaseConnection() as db:
    users = await db.execute("SELECT * FROM users")
```

### Async Generators

```python
async def fetch_pages(url: str) -> AsyncIterator[dict]:
    """Fetch paginated data lazily."""
    page = 1
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(f"{url}?page={page}")
            data = response.json()
            if not data["items"]:
                break
            yield data
            page += 1

# Usage
async for page in fetch_pages("https://api.example.com/items"):
    process_page(page)
```

### Async Best Practices

```python
# 1. Use asyncio.run() as entry point
async def main():
    await do_work()

asyncio.run(main())

# 2. Don't block the event loop
# Bad
async def bad_example():
    time.sleep(1)  # Blocks!

# Good
async def good_example():
    await asyncio.sleep(1)

# 3. Use run_in_executor for CPU-bound work
async def process_image(image_data: bytes) -> bytes:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Default executor
        cpu_intensive_resize,
        image_data,
    )
    return result

# 4. Use semaphores to limit concurrency
async def fetch_with_limit(urls: list[str], max_concurrent: int = 10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str):
        async with semaphore:
            async with httpx.AsyncClient() as client:
                return await client.get(url)

    return await asyncio.gather(*[fetch_one(url) for url in urls])

# 5. Handle cancellation gracefully
async def cancellable_task():
    try:
        while True:
            await do_work()
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await cleanup()
        raise
```

---

## 11. Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile a function
def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()

    # Code to profile
    result = expensive_operation()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(10)  # Top 10 functions

# Line profiler (install: pip install line_profiler)
# Add @profile decorator and run: kernprof -l -v script.py
@profile
def slow_function():
    result = []
    for i in range(10000):
        result.append(i ** 2)
    return result
```

### Memory Optimization

```python
# 1. Use generators for large datasets
# Bad - loads everything into memory
def get_all_lines(filename: str) -> list[str]:
    with open(filename) as f:
        return f.readlines()

# Good - yields one line at a time
def get_all_lines(filename: str) -> Iterator[str]:
    with open(filename) as f:
        for line in f:
            yield line

# 2. Use __slots__ for memory-efficient classes
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# 3. Use appropriate data structures
from collections import deque

# deque for fast append/pop from both ends
queue = deque(maxlen=1000)

# set for O(1) membership testing
seen = set()
if item not in seen:
    seen.add(item)
```

### Common Optimizations

```python
# 1. List comprehensions vs loops
# Slower
result = []
for i in range(1000):
    result.append(i ** 2)

# Faster
result = [i ** 2 for i in range(1000)]

# 2. Local variables vs global
# Slower
def slow():
    for _ in range(1000):
        x = global_var + 1

# Faster
def fast():
    local_var = global_var
    for _ in range(1000):
        x = local_var + 1

# 3. String concatenation
# Slower
result = ""
for s in strings:
    result += s

# Faster
result = "".join(strings)

# 4. Use built-in functions
# Slower
total = 0
for x in numbers:
    total += x

# Faster
total = sum(numbers)

# 5. Avoid repeated attribute lookups
# Slower
for item in items:
    self.container.data.append(item)

# Faster
append = self.container.data.append
for item in items:
    append(item)
```

### Caching

```python
from functools import lru_cache, cache

# LRU cache with size limit
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Unlimited cache (Python 3.9+)
@cache
def expensive_computation(x: int) -> int:
    return x ** 100

# Clear cache
fibonacci.cache_clear()

# Cache info
print(fibonacci.cache_info())
```

---

## 12. Security Best Practices

### Input Validation

```python
from pydantic import BaseModel, EmailStr, Field, validator

class UserCreate(BaseModel):
    """Validated user creation request."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)

    @validator("username")
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v

    @validator("password")
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        return v
```

### SQL Injection Prevention

```python
# Bad - SQL injection vulnerability
def get_user_bad(username: str):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)

# Good - parameterized query
def get_user_good(username: str):
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))

# Good - ORM (SQLAlchemy)
def get_user_orm(username: str):
    return session.query(User).filter(User.username == username).first()
```

### Password Handling

```python
import hashlib
import secrets

# Use dedicated library (recommended)
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Generate secure tokens
def generate_token(length: int = 32) -> str:
    return secrets.token_urlsafe(length)

def generate_api_key() -> str:
    return secrets.token_hex(32)
```

### Secrets Management

```python
import os
from dotenv import load_dotenv

# Never hardcode secrets
# Bad
API_KEY = "sk-1234567890abcdef"

# Good - environment variables
load_dotenv()
API_KEY = os.environ["API_KEY"]

# Good - with validation
def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Required environment variable {name} not set")
    return value

DATABASE_URL = get_required_env("DATABASE_URL")
```

### Path Traversal Prevention

```python
from pathlib import Path

def safe_file_read(base_dir: str, filename: str) -> str:
    """Safely read a file, preventing path traversal."""
    base = Path(base_dir).resolve()
    file_path = (base / filename).resolve()

    # Ensure the resolved path is within base directory
    if not str(file_path).startswith(str(base)):
        raise ValueError("Invalid file path")

    return file_path.read_text()
```

---

## 13. Modern Python Features

### Pattern Matching (Python 3.10+)

```python
def process_command(command: dict) -> str:
    match command:
        case {"action": "create", "name": name}:
            return f"Creating {name}"
        case {"action": "delete", "id": id_}:
            return f"Deleting {id_}"
        case {"action": "update", "id": id_, "data": data}:
            return f"Updating {id_} with {data}"
        case _:
            return "Unknown command"

# With guards
def categorize_number(n: int) -> str:
    match n:
        case _ if n < 0:
            return "negative"
        case 0:
            return "zero"
        case _ if n % 2 == 0:
            return "positive even"
        case _:
            return "positive odd"

# Matching classes
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

def describe_point(point: Point) -> str:
    match point:
        case Point(x=0, y=0):
            return "origin"
        case Point(x=0, y=y):
            return f"on y-axis at {y}"
        case Point(x=x, y=0):
            return f"on x-axis at {x}"
        case Point(x=x, y=y):
            return f"at ({x}, {y})"
```

### Walrus Operator (Python 3.8+)

```python
# Assign and use in one expression
if (n := len(data)) > 10:
    print(f"List is too long ({n} elements)")

# In comprehensions
results = [y for x in data if (y := expensive_computation(x)) > 0]

# In while loops
while (line := file.readline()):
    process(line)
```

### Data Classes

```python
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    tags: list[str] = field(default_factory=list)

    # Class variable (not an instance field)
    currency: ClassVar[str] = "USD"

    def total_value(self) -> float:
        return self.price * self.quantity

# Frozen (immutable) dataclass
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# With slots for memory efficiency
@dataclass(slots=True)
class User:
    name: str
    email: str
```

### f-strings and Advanced Formatting

```python
name = "Alice"
value = 123.456789

# Basic
print(f"Hello, {name}!")

# Expressions
print(f"Sum: {1 + 2}")

# Formatting
print(f"Value: {value:.2f}")  # 123.46
print(f"Padded: {name:>10}")  # "     Alice"
print(f"Binary: {42:b}")      # "101010"
print(f"Hex: {255:x}")        # "ff"

# Debug mode (Python 3.8+)
x = 10
print(f"{x=}")  # "x=10"
print(f"{x=:.2f}")  # "x=10.00"

# Nested expressions
print(f"{'left':<10} {'right':>10}")
```

### Context Variables (Python 3.7+)

```python
from contextvars import ContextVar

# Define context variable
request_id: ContextVar[str] = ContextVar("request_id", default="unknown")

async def handle_request(req_id: str):
    token = request_id.set(req_id)
    try:
        await process_request()
    finally:
        request_id.reset(token)

async def process_request():
    # Access context variable from anywhere in the call stack
    print(f"Processing request {request_id.get()}")
```

---

## 14. Code Quality Tools

### Ruff (Fast Linter and Formatter)

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "W",    # pycodestyle warnings
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "ARG",  # flake8-unused-arguments
    "PTH",  # flake8-use-pathlib
]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.lint.isort]
known-first-party = ["my_package"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### mypy (Type Checker)

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - pydantic

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
```

### Install and Run

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Run specific hook
ruff check .
ruff format .
mypy src/
```

---

## 15. Design Patterns and Principles

### SOLID Principles

```python
# Single Responsibility Principle
# Bad - class does too much
class UserManager:
    def create_user(self, data): ...
    def send_email(self, user): ...
    def generate_report(self, users): ...

# Good - separate responsibilities
class UserService:
    def create_user(self, data): ...

class EmailService:
    def send_email(self, recipient, message): ...

class ReportGenerator:
    def generate(self, data): ...


# Open/Closed Principle
# Open for extension, closed for modification
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process(self, amount: float) -> bool: ...

class CreditCardProcessor(PaymentProcessor):
    def process(self, amount: float) -> bool:
        # Credit card logic
        return True

class PayPalProcessor(PaymentProcessor):
    def process(self, amount: float) -> bool:
        # PayPal logic
        return True


# Dependency Inversion Principle
# Depend on abstractions, not concretions
class OrderService:
    def __init__(self, payment_processor: PaymentProcessor):
        self.payment_processor = payment_processor

    def checkout(self, order: Order) -> bool:
        return self.payment_processor.process(order.total)

# Inject dependency
service = OrderService(CreditCardProcessor())
```

### Factory Pattern

```python
from abc import ABC, abstractmethod

class Document(ABC):
    @abstractmethod
    def render(self) -> str: ...

class PDFDocument(Document):
    def render(self) -> str:
        return "PDF content"

class HTMLDocument(Document):
    def render(self) -> str:
        return "<html>content</html>"

class DocumentFactory:
    @staticmethod
    def create(doc_type: str) -> Document:
        match doc_type:
            case "pdf":
                return PDFDocument()
            case "html":
                return HTMLDocument()
            case _:
                raise ValueError(f"Unknown document type: {doc_type}")

# Usage
doc = DocumentFactory.create("pdf")
```

### Repository Pattern

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")

class Repository(ABC, Generic[T]):
    @abstractmethod
    def get(self, id: int) -> T | None: ...

    @abstractmethod
    def get_all(self) -> list[T]: ...

    @abstractmethod
    def add(self, entity: T) -> T: ...

    @abstractmethod
    def update(self, entity: T) -> T: ...

    @abstractmethod
    def delete(self, id: int) -> None: ...

class UserRepository(Repository[User]):
    def __init__(self, session: Session):
        self.session = session

    def get(self, id: int) -> User | None:
        return self.session.query(User).get(id)

    def get_all(self) -> list[User]:
        return self.session.query(User).all()

    def add(self, user: User) -> User:
        self.session.add(user)
        self.session.commit()
        return user

    def update(self, user: User) -> User:
        self.session.merge(user)
        self.session.commit()
        return user

    def delete(self, id: int) -> None:
        user = self.get(id)
        if user:
            self.session.delete(user)
            self.session.commit()
```

### Decorator Pattern

```python
from functools import wraps
import time

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    # May fail, will be retried
    return httpx.get(url).json()
```

### Context Manager Pattern

```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def timed_operation(name: str) -> Generator[None, None, None]:
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.3f} seconds")

# Usage
with timed_operation("data processing"):
    process_large_dataset()

# Class-based context manager
class DatabaseTransaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.begin()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        return False  # Don't suppress exceptions
```

---

## Quick Reference

### Must-Have Tools

| Tool | Purpose | Command |
|------|---------|---------|
| ruff | Linting + Formatting | `ruff check . && ruff format .` |
| mypy | Type checking | `mypy src/` |
| pytest | Testing | `pytest tests/` |
| pre-commit | Git hooks | `pre-commit run --all-files` |

### Project Checklist

- [ ] `pyproject.toml` configured
- [ ] Virtual environment set up
- [ ] Pre-commit hooks installed
- [ ] Type hints on public API
- [ ] Docstrings on public modules/classes/functions
- [ ] Tests with good coverage
- [ ] Logging configured
- [ ] Environment variables for secrets
- [ ] CI/CD pipeline configured

### Common Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Development
ruff check . --fix
ruff format .
mypy src/
pytest tests/ -v --cov

# Build
python -m build
twine check dist/*
```

---

## References

- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Real Python Tutorials](https://realpython.com/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
