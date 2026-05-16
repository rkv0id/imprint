"""Server-internal utilities. Not part of the public API."""

from __future__ import annotations

import os


def sqlite_file_path(store_url: str) -> str:
    """Resolve a sqlite store URL to a filesystem path.

    Accepts:
      sqlite:///abs/path  ->  /abs/path
      sqlite:///:memory:  ->  :memory:
      :memory:            ->  :memory:
      bare path           ->  path (with ~ expansion)

    ServerConfig.validate_sqlite_workers() already rejects Postgres URLs before
    this is called in SQLite-only code paths, so no URL-scheme validation needed.
    """
    if store_url == ":memory:":
        return ":memory:"
    if store_url.startswith("sqlite:///"):
        path = store_url[len("sqlite:///") :]
        if path == ":memory:":
            return ":memory:"
        return os.path.expanduser(path)
    return os.path.expanduser(store_url)
