"""Decide whether a range of commits changes anything the CI tests can see.

Usage: ``python code_changed.py BASE HEAD``

Prints one line per changed file with the reason it counts or not, then
``code=true`` or ``code=false``. When ``GITHUB_OUTPUT`` is set, the result is
also written there as the ``code`` step output.

A change counts as code unless every changed file is either

* matched by one of ``IGNORED_PATTERNS``, or
* a regular ``.py`` file, modified in place, whose syntax tree with docstrings
  removed is the same before and after (only docstrings, comments or layout
  changed).

The script fails safe: anything unexpected counts as code, so the tests run.
That covers a BASE or HEAD that is not a full commit SHA (manual runs, new
branches), a git error, a file that does not parse, a rename, a mode or type
change, and docstring changes while ``src/`` or ``tests/`` reads docstrings or
source text at runtime.

The CI workflow runs the copy of this script from the base commit, not the
one under review, so a pull request cannot change the rules that judge it.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

# Changes the tests cannot see.
IGNORED_PATTERNS = (
    "docs/**",
    "examples/**",
    "testdata/**",
    "**/*.md",
    ".github/workflows/cd.yml",
    ".github/workflows/format.yml",
    ".github/requirements/**",
    ".github/dependabot.yml",
)

# Only plain files can be compared as Python source; symlinks and submodules
# count as code.
_REGULAR_MODES = frozenset({"100644", "100755"})

# If the code under test can see docstrings or source text, docstring and
# comment changes can change test results.
_SOURCE_READERS = r"__doc__|getdoc|getsource|getcomments|getsourcelines|linecache"
_SOURCE_READER_PATHS = ("src", "tests")

_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
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


def _git(*args: str) -> bytes:
    # No pager, external diff or textconv: the output is git's own.
    return subprocess.run(
        ["git", "--no-pager", *args], check=True, capture_output=True
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


def same_code(old_source: bytes, new_source: bytes) -> bool:
    """True if the sources differ only in docstrings, comments or layout.

    The sources are parsed as bytes so that a coding declaration or BOM is
    honored exactly as the interpreter honors it: under ``# coding: utf-7``,
    text that reads as a comment in UTF-8 can be code.
    """
    old = strip_docstrings(ast.parse(old_source))
    new = strip_docstrings(ast.parse(new_source))
    return ast.dump(old, include_attributes=False) == ast.dump(
        new, include_attributes=False
    )


def _changes(base: str, head: str) -> list[tuple[str, str, str, list[str]]]:
    """Return ``(old_mode, new_mode, status, paths)`` for each changed file."""
    raw = _git(
        "diff", "--raw", "-z", "-M", "--no-color", "--no-ext-diff",
        "--no-textconv", "--no-abbrev", base, head, "--",
    ).decode("utf-8", "surrogateescape")  # fmt: skip
    fields = raw.split("\0")
    changes = []
    i = 0
    while i < len(fields) and fields[i]:
        old_mode, new_mode, _, _, status = fields[i].lstrip(":").split(" ")
        n_paths = 2 if status[0] in "RC" else 1
        changes.append((old_mode, new_mode, status[0], fields[i + 1 : i + 1 + n_paths]))
        i += 1 + n_paths
    return changes


def _file_reason(
    base: str, head: str, change: tuple[str, str, str, list[str]]
) -> str | None:
    """Return why this change counts as code, or None if it does not."""
    old_mode, new_mode, status, paths = change
    if all(is_ignored(p) for p in paths):
        return None
    path = paths[-1]
    if not path.endswith(".py"):
        return "non-Python file"
    if status != "M":
        return f"Python file with status {status}"
    if old_mode not in _REGULAR_MODES or new_mode not in _REGULAR_MODES:
        return f"mode {old_mode} -> {new_mode}"
    try:
        old = _git("cat-file", "blob", f"{base}:{path}")
        new = _git("cat-file", "blob", f"{head}:{path}")
        if same_code(old, new):
            return None
    except (SyntaxError, ValueError) as exc:
        return f"does not parse ({exc})"
    return "code differs"


def _source_readers(head: str) -> list[str]:
    """Files under src/ and tests/ that read docstrings or source at runtime."""
    result = subprocess.run(
        ["git", "--no-pager", "grep", "-l", "-E", _SOURCE_READERS, head, "--",
         *_SOURCE_READER_PATHS],
        capture_output=True, check=False,
    )  # fmt: skip
    if result.returncode not in {0, 1}:  # 1: no match
        msg = f"git grep failed: {result.stderr!r}"
        raise RuntimeError(msg)
    lines = result.stdout.decode("utf-8", "surrogateescape").splitlines()
    return [line.removeprefix(f"{head}:") for line in lines]


def code_changed(base: str, head: str) -> bool:
    for name, sha in (("BASE", base), ("HEAD", head)):
        if not _SHA.match(sha) or set(sha) == {"0"}:
            print(f"{name} is not a commit SHA ({sha!r}): treating as code change")
            return True
    changed = False
    docs_only_py = []
    for change in _changes(base, head):
        paths = change[3]
        reason = _file_reason(base, head, change)
        label = " -> ".join(paths)
        if reason is None:
            print(f"  skip  {label}")
            if paths[-1].endswith(".py") and not is_ignored(paths[-1]):
                docs_only_py.append(label)
        else:
            print(f"  CODE  {label} ({reason})")
            changed = True
    if docs_only_py and not changed and (readers := _source_readers(head)):
        print(f"  CODE  docstring/comment changes, and {', '.join(readers)} reads")
        print("        docstrings or source text at runtime")
        changed = True
    return changed


def main(argv: list[str]) -> int:
    base, head = [*argv, "", ""][:2]
    try:
        changed = code_changed(base, head)
    except Exception as exc:  # any failure must run the tests
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
