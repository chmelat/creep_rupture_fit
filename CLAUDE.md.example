# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working on this project.

## Quick Reference

### Environment

- **Use `python3` command** (not `python`) - system has no `python` symlink
- Python version: >=3.9
- Package manager: pip with pyproject.toml

### Project Structure

```
eis_analysis/              # Core package
  __init__.py              # Public API exports
  version.py               # Single source of truth for version
  cli/                     # CLI modules
  drt/                     # DRT analysis algorithms
  fitting/                 # Circuit fitting
  io/                      # Data loading (Gamry, CSV)
  validation/              # Kramers-Kronig validation
  visualization/           # Plotting functions
  analysis/                # Domain-specific analysis (oxide)
  rinf_estimation/         # R_inf estimation algorithms
  utils/                   # Shared utilities

eis.py                     # Main CLI entry point
tests/                     # Test suite (pytest)
doc/                       # Specialized documentation
example/                   # Example data files
```

### Common Commands

```bash
# Run CLI
python3 eis.py --help                    # Show help
python3 eis.py                           # Demo with synthetic data
python3 eis.py data.DTA                  # Analyze Gamry file
python3 eis.py data.DTA --lambda 0.5     # Manual regularization
python3 eis.py data.DTA --peak-method gmm  # GMM peak detection

# Testing
python3 -m pytest tests/                 # Run all tests
python3 -m pytest tests/test_drt.py -v   # Run specific test

# Code quality
ruff check eis_analysis/                 # Lint code
mypy eis_analysis/                       # Type check
```

### Input Data Formats

- **Gamry .DTA files** - Native format, auto-detected
- **CSV files** - Columns: `freq`, `Z_re`, `Z_im` (or `Zreal`, `Zimag`)
- **European CSV** - Semicolon separator, comma decimal (auto-detected)

### Dependencies

Core: `numpy>=1.20`, `scipy>=1.7`, `matplotlib>=3.4`

Dev: `ruff`, `mypy`, `pytest` (install with `pip install -e ".[dev]"`)

---

## Project Philosophy

### Goals

**EIS Analysis Toolkit** is a modular tool for electrochemical impedance spectroscopy analysis with emphasis on:

1. **Scientific accuracy** - Mathematically correct implementation of methods (DRT, GCV, GMM)
2. **Automation** - Minimizing manual tuning (auto-lambda, auto-circuit, GMM peaks)
3. **Reproducibility** - Objective, data-driven parameter choices
4. **Usability** - Easy to use for researchers (CLI + Python API)
5. **Modular design** - Clean separation of concerns, testability

### Design Principles

Standard best practices apply (DRY, Single Responsibility, YAGNI). Project-specific rules:

- **Monolithic files** - Keep under 500 lines, split if larger
- **Magic numbers** - All constants must be documented (why this value?)
- **Scientific accuracy over performance** - Correct first, fast second
- **Edge cases** - Always consider: empty data, NaN, single point, negative values

### Code Organization

**Separation rules:**
- CLI code separated from core logic (enables use as library)
- I/O separated from computations (testing without files)
- Visualization separated from algorithms (headless use)
- Domain logic isolated (extensibility for other domains)

---

## Testing

### Running Tests

```bash
python3 -m pytest tests/                 # All tests
python3 -m pytest tests/ -v              # Verbose output
python3 -m pytest tests/test_K_element.py  # Specific file
python3 -m pytest tests/ -k "voigt"      # Tests matching pattern
```

### Writing Tests

- Place tests in `tests/` directory
- Name files `test_*.py`
- Use synthetic data for reproducibility
- Test edge cases (empty, NaN, single point)
- Quantify correctness: `assert error < 0.05` (not "looks correct")

### Test Categories

- **Unit tests** - Individual functions
- **Integration tests** - Full workflow (`test_cli_integration.py`)
- **Regression tests** - Bug fixes (prevent recurrence)

---

## Code Quality

### Linting and Type Checking

```bash
ruff check eis_analysis/                 # Static analysis
ruff check eis_analysis/ --fix           # Auto-fix issues
mypy eis_analysis/                       # Type checking
```

### Code Style

- **Python:** PEP 8 (longer lines OK if more readable)
- **Docstrings:** NumPy style
- **Type hints:** Yes, but not dogmatically
- **Comments:** Explain "why", not "what"
- **Naming:** Descriptive (avoid abbreviations except: freq, Z, R, C, L)

---

## Documentation Organization

### Document Types

| Document | Content |
|----------|---------|
| README.md | User guide, theory, CLI reference, examples |
| CHANGELOG.md | Version history, migration guides |
| PYTHON_API.md | Python API reference for library usage |
| doc/*.md | Specialized guides (algorithms, math) |
| CLAUDE.md | Development guide (this file) |

### Single Source of Truth

- Version history -> CHANGELOG.md only
- Implementation details -> README.md only
- API reference -> docstrings in code
- Each piece of information has ONE primary location

### PDF Export Rules

**No emoji in documents** - DejaVu Serif font doesn't support them.

**Safe characters:** Standard ASCII, Greek letters (alpha, beta, lambda, omega), basic Unicode (x, +-, <=, >=, ->, deg, inf)

**Unsafe:** All emoji, box drawing characters, fancy quotes

---

## Version Management

**Single Source of Truth:** `eis_analysis/version.py`

**When releasing:**
1. Update `version.py`
2. Update CHANGELOG.md
3. Update documentation if needed

Details: See `VERSION_MANAGEMENT.md`

---

## Development Workflow

### Adding a New Feature

1. **Plan**
   - Backward compatible?
   - New module or extension?
   - Which layer? (I/O, algorithms, visualization, domain)

2. **Implement**
   - Choose correct module in core package
   - Follow existing API conventions
   - Add docstrings (NumPy style)
   - Add type hints

3. **Test**
   - Synthetic data (reproducible)
   - Real data (if available)
   - Edge cases

4. **Document**
   - Docstring (always)
   - README.md (if user-facing)
   - Specialized doc/*.md (if complex algorithm)

5. **Integrate**
   - Update CLI if user-facing
   - Update `__init__.py` exports
   - Update CHANGELOG.md
   - Test end-to-end workflow

### Bug Fix

1. Identify root cause
2. Write regression test (reproduces the bug)
3. Fix in appropriate module
4. Verify all tests pass
5. Update CHANGELOG.md
6. Commit: `fix(module): brief description`

### Git Workflow

**NEVER:**
- Commit to main without testing
- Force push to main
- Commit large binary files

**ALWAYS:**
- Descriptive commit messages
- Test before committing
- Update CHANGELOG.md
- One commit = one logical change

---

## Notes for AI Assistant

### When User Requests a Change

1. **Determine scope:**
   - Bugfix, feature, or refactoring?
   - Which modules affected?
   - Breaking change?

2. **Propose solution:**
   - Where to change code
   - What tests to add
   - What docs to update

3. **Implement:**
   - Core logic first
   - Integration second
   - Documentation third
   - Tests throughout

4. **Verify:**
   - Run tests: `python3 -m pytest tests/`
   - Check backward compatibility
   - Verify docs are current

### Prioritization

| Priority | Type |
|----------|------|
| High | Bugfixes affecting results, incorrect documentation |
| Medium | Performance improvements, new features |
| Low | Cosmetic changes, alternative implementations |

### Anti-patterns to Avoid

**Code:**
- Features "just in case" -> only what's requested
- Premature optimization -> correct first
- Ignoring edge cases -> test min/max/empty/NaN

**Documentation:**
- Duplicating information -> single source of truth
- Documentation without purpose -> every section must be useful

**Testing:**
- "Looks correct" -> quantify (error < X%)
- Happy path only -> test edge cases

---

**For implementation details see README.md. For API reference see docstrings and PYTHON_API.md.**
