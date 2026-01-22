# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working on this project.

## Quick Reference

### Environment

- **Use `python3` command** (not `python`) - system has no `python` symlink
- Dependencies: numpy, scipy, matplotlib

### Project Structure

```
crf.py                    # Main CLI
version.py                # Version (single source of truth)
models/
  lm.py                   # Larson-Miller model
  wsh.py                  # Wilshire model
example/                  # Sample data
```

### Common Commands

```bash
python3 crf.py example/data_T23.csv                           # LM fit
python3 crf.py example/data_T23.csv --model wsh \
    --tensile-data example/tensile_T23.csv                    # WSH fit
ruff check .              # Lint
mypy models/              # Type check
```

---

## Project Philosophy

### Goals

1. **Scientific accuracy** - Correct implementation of creep rupture models
2. **Extrapolation reliability** - Primary use case is long-term predictions
3. **Usability** - Minimal configuration, sensible defaults

### Design Principles

- **Separation of concerns** - Model logic in `models/`, CLI/I/O in main script
- **Scientific accuracy over performance** - Correct first, fast second
- **Magic numbers** - All constants must be documented (why this value?)

### Physical Constraints

When modifying model code:

- Stress must be positive [MPa]
- Temperature in Kelvin for calculations
- Time to rupture positive [h]
- Activation energy Q typically 100-400 kJ/mol

---

## Notes for AI Assistant

### When Making Changes

1. Model logic goes in `models/`
2. CLI/I/O stays in main script
3. Respect physical constraints
4. Test with example data after changes

### Anti-patterns to Avoid

- Wrong units (Celsius vs Kelvin, log10 vs ln)
- Silent failures - raise ValueError with clear message
- Breaking output format without reason

---

**For usage, theory, and CLI reference see README.md.**
