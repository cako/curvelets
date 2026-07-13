# Pull request

This PR bumps mypy's configured python_version to 3.14 so that mypy accepts modern numpy stubs
which use `Type` statement syntax introduced in Python 3.12+. This only affects static
type checking and does not change runtime requirements.

The change prevents pre-commit's mypy hook from failing on CI when parsing numpy
stub files.
