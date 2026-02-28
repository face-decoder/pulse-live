# Python Code Review Checklist

A comprehensive checklist for conducting effective Python code reviews. Use this guide to ensure code quality, maintainability, security, and performance.

---

## Table of Contents

1. [Style and Formatting](#1-style-and-formatting)
2. [Naming Conventions](#2-naming-conventions)
3. [Function and Method Design](#3-function-and-method-design)
4. [Class and Object Design](#4-class-and-object-design)
5. [Error Handling](#5-error-handling)
6. [Security](#6-security)
7. [Performance](#7-performance)
8. [Testing](#8-testing)
9. [Documentation](#9-documentation)
10. [Code Organization](#10-code-organization)
11. [Dependencies](#11-dependencies)
12. [Pre-Production Cleanup](#12-pre-production-cleanup)

---

## 1. Style and Formatting

Consistent formatting improves readability and reduces cognitive load for team members.

### Automated Checks

- [ ] **Linter passes** - Code passes `ruff`, `flake8`, or `pylint` without errors
- [ ] **Formatter applied** - Code is formatted with `black`, `ruff format`, or similar
- [ ] **Import sorting** - Imports are sorted (standard library, third-party, local)
- [ ] **Type checker passes** - No `mypy` or `pyright` errors

### Manual Checks

- [ ] **Line length** - Lines are under 88-120 characters (project standard)
- [ ] **Blank lines** - Appropriate spacing between functions/classes (2 lines top-level, 1 line methods)
- [ ] **Indentation** - Consistent 4-space indentation throughout
- [ ] **No trailing whitespace** - Lines don't end with unnecessary spaces
- [ ] **File ends with newline** - All files end with a single newline

### Common Issues to Flag

```python
# Bad: Inconsistent spacing
def foo( x,y ):
    return x+y

# Good: Consistent spacing
def foo(x, y):
    return x + y

# Bad: Multiple statements on one line
if condition: do_something(); do_another()

# Good: Separate lines
if condition:
    do_something()
    do_another()
```

---

## 2. Naming Conventions

Clear, descriptive names make code self-documenting and easier to maintain.

### Checklist

- [ ] **Variables use snake_case** - `user_count`, `is_valid`, `max_retries`
- [ ] **Functions use snake_case** - `calculate_total()`, `get_user_by_id()`
- [ ] **Classes use PascalCase** - `UserAccount`, `HTTPClient`, `XMLParser`
- [ ] **Constants use UPPER_SNAKE_CASE** - `MAX_CONNECTIONS`, `DEFAULT_TIMEOUT`
- [ ] **Private members use leading underscore** - `_internal_method()`, `_cache`
- [ ] **No single-letter names** (except loop counters `i`, `j`, `k`)
- [ ] **No ambiguous abbreviations** - `usr` → `user`, `cnt` → `count`
- [ ] **Boolean variables are questions** - `is_active`, `has_permission`, `can_edit`

### Naming Quality Questions

- [ ] Can you understand what this variable holds without reading surrounding code?
- [ ] Does the function name describe what it does (verb + noun)?
- [ ] Are similar concepts named consistently throughout the codebase?
- [ ] Would a new team member understand these names?

### Examples

```python
# Bad: Unclear names
def proc(d):
    r = []
    for x in d:
        if x.s == 1:
            r.append(x)
    return r

# Good: Descriptive names
def get_active_users(users: list[User]) -> list[User]:
    active_users = []
    for user in users:
        if user.status == UserStatus.ACTIVE:
            active_users.append(user)
    return active_users

# Better: Using comprehension
def get_active_users(users: list[User]) -> list[User]:
    return [user for user in users if user.status == UserStatus.ACTIVE]
```

---

## 3. Function and Method Design

Small, single-purpose functions are easier to understand, test, and maintain.

### Single Responsibility

- [ ] **Does one thing** - Function has a single, clear purpose
- [ ] **Fits on one screen** - Generally under 20-30 lines
- [ ] **Single level of abstraction** - Doesn't mix high-level and low-level operations
- [ ] **No side effects** (unless clearly documented)
- [ ] **Returns consistent types** - Doesn't return different types conditionally

### Parameters

- [ ] **Few parameters** - Ideally 3 or fewer; consider a dataclass for more
- [ ] **No boolean flags** - Prefer separate functions over `do_thing(full=True)`
- [ ] **Sensible defaults** - Default values are safe and commonly used
- [ ] **Type hints present** - All parameters and return types annotated
- [ ] ] **No mutable default arguments** - Don't use `def foo(items=[])`

### Common Issues

```python
# Bad: Mutable default argument
def add_item(item, items=[]):  # Bug: list is shared!
    items.append(item)
    return items

# Good: Use None and create new list
def add_item(item, items: list | None = None) -> list:
    if items is None:
        items = []
    items.append(item)
    return items

# Bad: Boolean flag creating two functions in one
def get_users(include_inactive=False):
    if include_inactive:
        return db.query(User).all()
    return db.query(User).filter(User.active == True).all()

# Good: Separate, clear functions
def get_all_users() -> list[User]:
    return db.query(User).all()

def get_active_users() -> list[User]:
    return db.query(User).filter(User.active == True).all()
```

### Loop Alternatives

- [ ] **Consider list comprehensions** - More Pythonic for simple transformations
- [ ] **Consider generator expressions** - For memory-efficient iteration
- [ ] **Consider built-in functions** - `map()`, `filter()`, `sum()`, `any()`, `all()`
- [ ] **Avoid nested loops** - Extract to separate functions when deep

```python
# Bad: Verbose loop
result = []
for item in items:
    if item.is_valid:
        result.append(item.value)

# Good: List comprehension
result = [item.value for item in items if item.is_valid]

# Bad: Nested loop for checking
found = False
for item in items:
    if item.matches(criteria):
        found = True
        break

# Good: Use any()
found = any(item.matches(criteria) for item in items)
```

---

## 4. Class and Object Design

### Design Principles

- [ ] **Single Responsibility** - Class has one reason to change
- [ ] **Proper encapsulation** - Implementation details are private
- [ ] **Favor composition over inheritance** - Use has-a, not is-a when appropriate
- [ ] **Design patterns used correctly** - Singleton, Factory, etc. implemented properly
- [ ] **Dataclasses for data containers** - Use `@dataclass` for DTOs and value objects

### Immutability

- [ ] **Prefer immutable types** - Easier to reason about, fewer bugs
- [ ] **Use frozen dataclasses** - `@dataclass(frozen=True)` for value objects
- [ ] **Avoid modifying input parameters** - Create new objects instead
- [ ] **Use tuples over lists** when data shouldn't change

```python
# Bad: Mutable class with side effects
class Config:
    def __init__(self):
        self.settings = {}

    def load(self, path):
        self.settings = load_file(path)  # Mutates state

# Good: Immutable configuration
@dataclass(frozen=True)
class Config:
    database_url: str
    debug: bool
    max_connections: int

    @classmethod
    def from_file(cls, path: str) -> "Config":
        data = load_file(path)
        return cls(**data)
```

### Class Checklist

- [ ] **`__init__` only initializes** - No complex logic in constructor
- [ ] **`__repr__` defined** - For debugging (auto with dataclass)
- [ ] **`__eq__` defined if needed** - For comparison (auto with dataclass)
- [ ] **No god objects** - Classes shouldn't do everything
- [ ] **Interfaces are small** - Prefer many small interfaces to one large

---

## 5. Error Handling

### Exception Handling

- [ ] **Specific exceptions caught** - Never bare `except:` or `except Exception:`
- [ ] **Exceptions not silenced** - Don't catch and ignore without logging
- [ ] **Custom exceptions defined** - For domain-specific errors
- [ ] **Exception chain preserved** - Use `raise ... from e`
- [ ] **Try blocks are small** - Only wrap code that might fail
- [ ] **Finally for cleanup** - Or use context managers

### Common Issues

```python
# Bad: Bare except silences everything
try:
    process_data()
except:
    pass

# Bad: Catching too broad, losing information
try:
    user = get_user(user_id)
    orders = get_orders(user.id)
except Exception as e:
    return None  # Which operation failed?

# Good: Specific handling with context
try:
    user = get_user(user_id)
except UserNotFoundError:
    logger.warning(f"User {user_id} not found")
    return None
except DatabaseError as e:
    logger.exception("Database error fetching user")
    raise ServiceUnavailableError("Database unavailable") from e
```

### Error Messages

- [ ] **Actionable messages** - Tell user/developer what to do
- [ ] **Include context** - What values caused the error?
- [ ] **No sensitive data** - Don't expose passwords, tokens in errors
- [ ] **Consistent format** - Similar errors formatted similarly

```python
# Bad: Unhelpful error
raise ValueError("Invalid input")

# Good: Actionable with context
raise ValueError(
    f"Invalid email format: '{email}'. "
    f"Expected format: user@domain.com"
)
```

---

## 6. Security

### Input Validation

- [ ] **All external inputs validated** - User input, API responses, file contents
- [ ] **Type checking** - Verify types before processing
- [ ] **Range checking** - Numbers within expected bounds
- [ ] **Format validation** - Emails, URLs, dates properly formatted
- [ ] **Length limits** - Strings and collections have max sizes

### SQL and Injection

- [ ] **No string formatting in SQL** - Use parameterized queries only
- [ ] **ORM used correctly** - No raw SQL concatenation
- [ ] **Input sanitized** - Even with ORM, validate inputs

```python
# CRITICAL: SQL Injection vulnerability
query = f"SELECT * FROM users WHERE id = {user_id}"  # NEVER DO THIS

# Safe: Parameterized query
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# Safe: ORM
user = session.query(User).filter(User.id == user_id).first()
```

### Secrets and Sensitive Data

- [ ] **No hardcoded secrets** - Use environment variables
- [ ] **Secrets not logged** - Check log statements don't expose credentials
- [ ] **Secrets not in URLs** - API keys not in query strings
- [ ] **`.env` in `.gitignore`** - Environment files not committed
- [ ] **Passwords properly hashed** - Using bcrypt, argon2, or similar

### Path Traversal

- [ ] **User paths validated** - Prevent `../` attacks
- [ ] **Paths resolved and checked** - Verify within allowed directory

```python
# Bad: Path traversal vulnerability
def read_file(filename):
    return open(f"/data/{filename}").read()  # User could pass "../etc/passwd"

# Good: Validate path
def read_file(filename: str) -> str:
    base = Path("/data").resolve()
    file_path = (base / filename).resolve()
    if not str(file_path).startswith(str(base)):
        raise ValueError("Invalid file path")
    return file_path.read_text()
```

### Other Security Checks

- [ ] **No `eval()` or `exec()`** - Especially with user input
- [ ] **No `pickle` with untrusted data** - Use JSON instead
- [ ] **HTTPS for external calls** - No HTTP for sensitive data
- [ ] **Rate limiting considered** - For public endpoints
- [ ] **Authentication checked** - Protected resources require auth

---

## 7. Performance

### Algorithmic Efficiency

- [ ] **Appropriate data structures** - Set for membership, dict for lookup
- [ ] **No unnecessary iterations** - Don't loop multiple times when once suffices
- [ ] **Early returns** - Exit loops/functions when result is known
- [ ] **Lazy evaluation** - Use generators for large datasets

### Common Performance Issues

```python
# Bad: O(n) lookup in list
if user_id in user_list:  # Scans entire list
    process(user_id)

# Good: O(1) lookup in set
user_set = set(user_list)
if user_id in user_set:  # Constant time
    process(user_id)

# Bad: Loading entire file into memory
data = open("huge_file.txt").read()
for line in data.split("\n"):
    process(line)

# Good: Streaming line by line
with open("huge_file.txt") as f:
    for line in f:
        process(line)

# Bad: Repeated string concatenation
result = ""
for item in items:
    result += str(item)  # Creates new string each time

# Good: Join once
result = "".join(str(item) for item in items)
```

### Database Queries

- [ ] **N+1 queries avoided** - Use eager loading or batch queries
- [ ] **Indexes used** - Queries use indexed columns
- [ ] **Pagination implemented** - Large result sets paginated
- [ ] **Connections pooled** - Not creating new connections per request

### Caching

- [ ] **Appropriate caching** - Expensive operations cached when beneficial
- [ ] **Cache invalidation** - Cache is cleared when data changes
- [ ] **TTL set** - Cached data expires appropriately

```python
from functools import lru_cache

# Good: Cache expensive computation
@lru_cache(maxsize=128)
def get_user_permissions(user_id: int) -> set[str]:
    # Expensive database query
    return fetch_permissions_from_db(user_id)
```

---

## 8. Testing

### Test Coverage

- [ ] **Happy path tested** - Normal usage works
- [ ] **Edge cases tested** - Empty inputs, boundaries, nulls
- [ ] **Error cases tested** - Expected exceptions raised
- [ ] **Integration points tested** - External services mocked appropriately

### Test Quality

- [ ] **Tests are independent** - No shared mutable state
- [ ] **Tests are deterministic** - Same result every time
- [ ] **Test names describe behavior** - `test_user_creation_with_invalid_email_raises_error`
- [ ] **One assertion focus** - Each test verifies one behavior
- [ ] **Fixtures used appropriately** - Shared setup in fixtures

### Test Checklist

```python
# Good test structure
class TestUserService:
    """Tests for UserService."""

    def test_create_user_with_valid_data_succeeds(self, db):
        """Creating a user with valid data should succeed."""
        service = UserService(db)
        user = service.create(name="Alice", email="alice@example.com")

        assert user.id is not None
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_create_user_with_duplicate_email_raises_error(self, db, existing_user):
        """Creating a user with existing email should raise DuplicateEmailError."""
        service = UserService(db)

        with pytest.raises(DuplicateEmailError) as exc_info:
            service.create(name="Bob", email=existing_user.email)

        assert "already exists" in str(exc_info.value)

    def test_create_user_with_empty_name_raises_validation_error(self, db):
        """Creating a user with empty name should raise ValidationError."""
        service = UserService(db)

        with pytest.raises(ValidationError):
            service.create(name="", email="test@example.com")
```

### What Should Be Tested?

- [ ] Public API methods
- [ ] Business logic and calculations
- [ ] Data transformations
- [ ] Error handling paths
- [ ] Edge cases (empty, null, max values)
- [ ] Integration with external services (mocked)

---

## 9. Documentation

### Docstrings

- [ ] **Public functions documented** - Args, returns, raises
- [ ] **Public classes documented** - Purpose and usage
- [ ] **Modules have docstrings** - Overview at top of file
- [ ] **Complex logic explained** - Why, not just what
- [ ] **Examples included** - For non-obvious usage

### Docstring Quality

```python
# Bad: Useless docstring
def process(data):
    """Process the data."""
    pass

# Good: Informative docstring
def process_transaction(
    transaction: Transaction,
    validate: bool = True,
) -> TransactionResult:
    """Process a financial transaction and update account balances.

    Validates the transaction, checks sufficient funds, applies the
    transaction to source and destination accounts, and records
    the transaction in the audit log.

    Args:
        transaction: The transaction to process.
        validate: Whether to validate before processing. Set to False
            only for internal transfers that are pre-validated.

    Returns:
        TransactionResult containing the updated balances and
        confirmation number.

    Raises:
        InsufficientFundsError: If source account lacks funds.
        InvalidTransactionError: If validation fails.
        AccountLockedError: If either account is locked.

    Example:
        >>> txn = Transaction(from_account=123, to_account=456, amount=100)
        >>> result = process_transaction(txn)
        >>> print(result.confirmation_number)
        'TXN-2024-001234'
    """
```

### Comments

- [ ] **No obvious comments** - Don't explain what code does literally
- [ ] **Why comments present** - Explain non-obvious decisions
- [ ] **TODO/FIXME with context** - Include ticket numbers or deadlines
- [ ] **No commented-out code** - Delete it; use version control

```python
# Bad: Obvious comment
# Increment counter by 1
counter += 1

# Good: Explains why
# Use exponential backoff to avoid overwhelming the server
# during recovery after an outage
delay = base_delay * (2 ** attempt)

# Good: TODO with context
# TODO(JIRA-123): Replace with batch API once available
for item in items:
    api.process_single(item)
```

---

## 10. Code Organization

### Imports

- [ ] **Absolute imports preferred** - `from mypackage.module import func`
- [ ] **No wildcard imports** - Don't use `from module import *`
- [ ] **Imports at top of file** - Not inside functions (except for circular deps)
- [ ] **No unused imports** - Remove imports that aren't used
- [ ] **Grouped correctly** - stdlib, third-party, local (with blank lines)

```python
# Good: Properly organized imports
import os
import sys
from pathlib import Path

import httpx
from pydantic import BaseModel

from mypackage.config import settings
from mypackage.utils import helpers
```

### Module Structure

- [ ] **One class per file** (for large classes) or related classes grouped
- [ ] **Logical grouping** - Related functionality in same module
- [ ] **No circular imports** - Restructure if present
- [ ] **`__all__` defined** - Explicit public API in modules

### DRY (Don't Repeat Yourself)

- [ ] **No duplicated code** - Extract common logic to functions
- [ ] **Utilities extracted** - Reusable code in utils modules
- [ ] **Built-ins used** - Don't reimplement standard library
- [ ] **Third-party considered** - Use maintained packages for common needs

```python
# Bad: Duplicated logic
def process_user(user):
    if not user.email or "@" not in user.email:
        raise ValueError("Invalid email")
    # ... process

def process_admin(admin):
    if not admin.email or "@" not in admin.email:
        raise ValueError("Invalid email")
    # ... process

# Good: Extracted validation
def validate_email(email: str) -> None:
    if not email or "@" not in email:
        raise ValueError(f"Invalid email: {email}")

def process_user(user):
    validate_email(user.email)
    # ... process

def process_admin(admin):
    validate_email(admin.email)
    # ... process
```

---

## 11. Dependencies

### Dependency Management

- [ ] **Dependencies pinned** - Exact versions in lock file
- [ ] **Minimal dependencies** - Only add what's necessary
- [ ] **Maintained packages** - No abandoned/unmaintained deps
- [ ] **Security vulnerabilities checked** - Run `pip-audit` or similar
- [ ] **License compatible** - Dependencies have compatible licenses

### Upgrading Dependencies

- [ ] **Changelog reviewed** - Check for breaking changes
- [ ] **Tests pass** - After upgrade, all tests still pass
- [ ] **Gradual upgrades** - Don't upgrade everything at once
- [ ] **Major versions careful** - Plan and test major version bumps

### Import Best Practices

- [ ] **No version-specific hacks** - Use proper version constraints
- [ ] **Optional dependencies handled** - Try/except for optional imports
- [ ] **Import errors graceful** - Clear message if dependency missing

```python
# Good: Optional dependency handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

def export_to_dataframe(data):
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for this feature. "
            "Install with: pip install mypackage[pandas]"
        )
    return pd.DataFrame(data)
```

---

## 12. Pre-Production Cleanup

### Remove Before Merging

- [ ] **No debug statements** - Remove `print()`, `breakpoint()`, `pdb`
- [ ] **No commented-out code** - Delete unused code
- [ ] **No TODO for this PR** - Complete or convert to tickets
- [ ] **No temporary files** - Remove test files, scratch code
- [ ] **No hardcoded values** - URLs, IDs should be configurable

### Verify Before Merging

- [ ] **All tests pass** - CI is green
- [ ] **No merge conflicts** - Rebased on main
- [ ] **Commit messages clean** - Descriptive, follows conventions
- [ ] **PR description complete** - Explains what and why
- [ ] **No secrets committed** - Check for API keys, passwords

### Final Checks

```python
# Remove these before production:
print("DEBUG:", some_value)  # Remove
breakpoint()  # Remove
import pdb; pdb.set_trace()  # Remove

# # Old implementation  # Remove commented code
# def old_function():
#     pass

# TODO: Fix this later  # Convert to ticket or fix now

API_KEY = "sk-1234567890"  # NEVER commit secrets
```

---

## Quick Reference Card

### Before Starting Review

1. Pull the branch and run tests locally
2. Run linter: `ruff check .`
3. Run type checker: `mypy src/`
4. Read the PR description

### During Review

Ask yourself:
- Would I understand this code in 6 months?
- Is this the simplest solution?
- Are there obvious bugs or edge cases?
- Is it secure?
- Is it tested?

### Red Flags to Watch For

| Issue | Why It's Bad |
|-------|--------------|
| Bare `except:` | Hides bugs, catches KeyboardInterrupt |
| Mutable default args | Shared state between calls |
| `eval()`/`exec()` | Security vulnerability |
| SQL string formatting | Injection vulnerability |
| No type hints | Harder to maintain |
| Magic numbers | Unclear meaning |
| God functions (>50 lines) | Hard to test and maintain |
| No tests | Can't verify behavior |

### Approve When

- [ ] Code is correct and handles edge cases
- [ ] Code is readable and maintainable
- [ ] Tests cover important paths
- [ ] No security issues
- [ ] Follows project conventions
- [ ] Documentation is adequate

---

## References

- [JetBrains Python Code Review Checklist](https://www.jetbrains.com/pages/static-code-analysis-guide/python-code-review-checklist/)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [OWASP Python Security](https://cheatsheetseries.owasp.org/cheatsheets/Python_Security_Cheat_Sheet.html)
- [Real Python Code Quality](https://realpython.com/python-code-quality/)
