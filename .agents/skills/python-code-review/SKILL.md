---
name: python-code-review
description: Review Python code changes in merge requests and pull requests. Use when asked to review Python code, MR/PR reviews, code quality checks, or when phrases like "review this code", "check this PR", "review my changes", "review mr on this branch", "review MR", "review PR", "PR reivew", "MR review" or "code review" are mentioned for Python files.
---

# Python Code Review

Review Python code changes systematically using established best practices and checklists.

## Review Workflow

### 1. Gather Context

```bash
# Get the diff to review
git diff main...HEAD

# Or for a specific PR
gh pr diff <pr-number>
```

### 2. Run Automated Checks First

Before manual review, verify automated tools pass:

```bash
ruff check .                    # Linting
ruff format --check .           # Formatting
mypy src/                       # Type checking
pytest tests/ -v                # Tests
```

### 3. Systematic Code Review

Review each file in the diff, checking these categories in order:

**Security (Critical)**
- No SQL string formatting (use parameterized queries)
- No `eval()`/`exec()` with user input
- No hardcoded secrets
- Inputs validated at boundaries
- No path traversal vulnerabilities

**Correctness**
- Logic handles edge cases (empty, None, boundaries)
- Error handling is specific (no bare `except:`)
- Exception chains preserved (`raise ... from e`)
- No mutable default arguments
- Types are correct and complete

**Design**
- Functions do one thing (<30 lines)
- Classes follow single responsibility
- No code duplication (DRY)
- Appropriate data structures used
- No premature optimization

**Style**
- Naming is clear and consistent (snake_case, PascalCase)
- No magic numbers/strings
- Docstrings on public API
- No commented-out code

### 4. Provide Feedback

Structure feedback by severity:

1. **Blockers** - Must fix before merge (security, bugs, breaking changes)
2. **Suggestions** - Should consider (design, performance, readability)
3. **Nits** - Minor style issues (optional to fix)

## Red Flags to Catch

| Pattern | Issue |
|---------|-------|
| `except:` or `except Exception:` | Catches too much, hides bugs |
| `def foo(items=[])` | Mutable default argument bug |
| `f"SELECT * FROM {table}"` | SQL injection |
| `eval(user_input)` | Code injection |
| `password = "secret123"` | Hardcoded secret |
| No type hints on public functions | Harder to maintain |
| Functions >50 lines | Likely doing too much |
| `# TODO` without ticket | Tech debt without tracking |

## Common Patterns to Suggest

```python
# Instead of bare except
try:
    process()
except:  # Bad
    pass

# Use specific exceptions
try:
    process()
except ValueError as e:
    logger.exception("Processing failed")
    raise

# Instead of mutable defaults
def add_item(item, items=[]):  # Bad
    items.append(item)

# Use None pattern
def add_item(item, items: list | None = None) -> list:
    if items is None:
        items = []
    items.append(item)
    return items

# Instead of verbose loops
result = []
for x in data:
    if x.valid:
        result.append(x.value)

# Use comprehensions
result = [x.value for x in data if x.valid]
```

## Reference Guides

For detailed guidelines, consult:

- **[Checklist](references/checklist.md)** - Complete review checklist with checkboxes
- **[Best Practices](references/best-practices.md)** - Comprehensive Python best practices

Use these references when:
- Explaining why something should change
- Providing examples of better patterns
- Checking specific categories in depth
