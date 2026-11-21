"""Decide whether a range of commits changes anything the CI tests can see.

Usage: ``python code_changed.py BASE HEAD``

Prints one line per changed file with the reason it counts or not, then
``code=true`` or ``code=false``. When ``GITHUB_OUTPUT`` is set, the result is
also written there as the ``code`` step output.

A change counts as code unless every changed file is either

* matched by one of the CI workflow's ``paths-ignore`` patterns, or
* an existing ``.py`` file whose syntax tree, with docstrings removed, is the
  same before and after (only docstrings, comments or formatting changed).

Anything unexpected (an empty or all-zero BASE, a missing commit, a git
error, a file that does not parse) counts as code, so the tests run.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

# Keep in sync with `paths-ignore` in .github/workflows/ci.yml.
IGNORED_PATTERNS = (
    "docs/**",
    "**/*.md",
    ".github/workflows/cd.yml",
    ".github/workflows/format.yml",
    ".github/requirements/**",
    ".github/dependabot.yml",
)

_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a GitHub Actions path glob (``*``, ``**``) to a regex."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


_IGNORED = tuple(_glob_to_regex(p) for p in IGNORED_PATTERNS)


def is_ignored(path: str) -> bool:
    return any(regex.match(path) for regex in _IGNORED)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstrings in place; an emptied body becomes ``pass``."""
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def same_code(old_source: str, new_source: str) -> bool:
    """True if the sources differ only in docstrings, comments or layout."""
    old = strip_docstrings(ast.parse(old_source))
    new = strip_docstrings(ast.parse(new_source))
    return ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    )


def _file_reason(base: str, head: str, status: str, paths: list[str]) -> str | None:
    """Return why this change counts as code, or None if it does not."""
    if all(is_ignored(p) for p in paths):
        return None
    path = paths[-1]
    if not path.endswith(".py"):
        return "non-Python file"
    if status != "M":
        return f"Python file with status {status}"
    try:
        old = _git("show", f"{base}:{path}")
        new = _git("show", f"{head}:{path}")
        if same_code(old, new):
            return None
    except SyntaxError as exc:
        return f"does not parse ({exc.msg}, line {exc.lineno})"
    return "code differs"


def code_changed(base: str, head: str) -> bool:
    if not base or set(base) == {"0"}:
        print(f"no base commit ({base!r}): treating as code change")
        return True
    diff = _git("diff", "--name-status", "-M", "--no-color", base, head)
    changed = False
    for line in diff.splitlines():
        status, *paths = line.split("\t")
        reason = _file_reason(base, head, status[0], paths)
        label = " -> ".join(paths)
        if reason is None:
            print(f"  skip  {label}")
        else:
            print(f"  CODE  {label} ({reason})")
            changed = True
    return changed


def main(argv: list[str]) -> int:
    base, head = [*argv, "", ""][:2]
    try:
        changed = code_changed(base, head or "HEAD")
    except Exception as exc:
        print(f"error ({exc!r}): treating as code change")
        changed = True
    result = f"code={'true' if changed else 'false'}"
    print(result)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as fh:  # noqa: PTH123
            fh.write(result + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
