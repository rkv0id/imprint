"""ASCII audit: every text file in the repo must be pure ASCII.

We deliberately reject unicode AI-tells (em dashes, smart quotes, unicode
arrows, ellipsis, box-drawing) anywhere in the codebase or documentation.
ASCII-only means a contributor can grep, copy, and review without ever seeing
a character that wasn't typed deliberately.

This test enforces the rule. Auto-generated files (uv.lock) and license text
are excluded.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXTENSIONS_TO_CHECK = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
    ".cfg",
    ".ini",
    ".example",
}
NAMED_FILES_TO_CHECK = {"justfile", ".gitignore", ".env.example", ".python-version"}
FILES_TO_SKIP = {"uv.lock", "LICENSE"}
DIRS_TO_SKIP = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".pyright_cache",
    ".git",
}


def _should_check(path: Path) -> bool:
    if path.name in FILES_TO_SKIP:
        return False
    if path.name in NAMED_FILES_TO_CHECK:
        return True
    return path.suffix in EXTENSIONS_TO_CHECK


def test_repo_is_ascii_only() -> None:
    violations: list[tuple[Path, int, str]] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in DIRS_TO_SKIP for part in path.parts):
            continue
        if not _should_check(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append((path, 0, "<not utf-8>"))
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            try:
                line.encode("ascii")
            except UnicodeEncodeError:
                rel = path.relative_to(REPO_ROOT)
                violations.append((rel, line_no, line.strip()))
                break  # one report per file is enough

    if violations:
        msg = "Non-ASCII content found:\n" + "\n".join(
            f"  {path}:{line_no}: {snippet}" for path, line_no, snippet in violations
        )
        raise AssertionError(msg)
