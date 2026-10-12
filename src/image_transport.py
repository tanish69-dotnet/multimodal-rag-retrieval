"""Trusted-root image transport (Phase 9N).

Reads figure bytes ONLY from explicitly allowed roots, enforces byte +
encoded-payload caps, and returns base64 for figure evidence. Anything
outside the roots — traversal, UNC, absolute paths elsewhere, symlinks
escaping the roots — yields ``(None, None)`` so retrieval degrades to
caption/context instead of failing text/table paths.

No network, no GPU, no new model. Pure stdlib + Pillow (already pinned).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

# Raw figure bytes cap (10 MiB) and encoded payload cap (~13.4M chars).
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_B64_CHARS = (MAX_IMAGE_BYTES * 4) // 3 + 64

_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_ALLOWED_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def trusted_roots(extra: "str | Path | None" = None) -> list[Path]:
    """Explicitly allowed image roots (resolved, no guessing).

    - ``data/source_artifacts/figures`` (versioned restored figures)
    - the OS temp dir subtree ``p2_ingest/`` (per-ingest figure artifacts)
    - ``extra`` only when the caller passes an explicit directory
    """
    roots = [Path("data/source_artifacts/figures").resolve()]
    try:
        import tempfile

        roots.append((Path(tempfile.gettempdir()) / "p2_ingest").resolve())
    except Exception:
        pass
    if extra:
        try:
            roots.append(Path(extra).resolve())
        except Exception:
            pass
    return roots


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_trusted_image(path_str: str | None, roots: list[Path] | None = None) -> Optional[Path]:
    """Resolve ``path_str`` strictly inside one of ``roots``.

    Returns the resolved file path, or ``None`` for: empty input, UNC paths,
    traversal/absolute paths outside every root, symlinks escaping the roots,
    missing files, or non-image suffixes.
    """
    if not path_str or not isinstance(path_str, str):
        return None
    stripped = path_str.strip()
    if not stripped:
        return None
    if stripped.startswith("\\\\") or stripped.startswith("//"):
        return None
    roots = roots if roots is not None else trusted_roots()
    candidate = (Path.cwd() / stripped).resolve() if not Path(stripped).is_absolute() else Path(stripped).resolve()
    # Reject symlink escapes: the resolved path must stay within a root AND
    # no path component may be a symlink pointing outside (resolve() already
    # collapses symlinks, so the within-check below covers it).
    if not any(_is_within(root, candidate) for root in roots):
        return None
    if candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
        return None
    try:
        if not candidate.is_file():
            return None
    except OSError:
        return None
    return candidate


def image_to_b64(
    path_str: str | None, roots: list[Path] | None = None
) -> tuple[Optional[str], Optional[str]]:
    """Read trusted figure bytes → ``(b64, mime)`` or ``(None, None)``.

    Caps: raw bytes ≤ ``MAX_IMAGE_BYTES``; encoded ≤ ``MAX_B64_CHARS``.
    Malformed/unreadable files degrade to ``(None, None)``.

    Absolute Windows paths are accepted here *only* when they resolve inside
    an explicit trusted root (single-machine demo). Anything else — including
    pytest tmp dirs that no longer exist — degrades to ``(None, None)``.
    """
    resolved = resolve_trusted_image(path_str, roots)
    if resolved is None:
        return None, None
    try:
        if resolved.stat().st_size > MAX_IMAGE_BYTES:
            return None, None
        raw = resolved.read_bytes()
    except OSError:
        return None, None
    if len(raw) > MAX_IMAGE_BYTES or not raw:
        return None, None
    mime = _ALLOWED_MIMES.get(resolved.suffix.lower())
    if mime is None:
        return None, None
    try:
        encoded = base64.b64encode(raw).decode("ascii")
    except Exception:
        return None, None
    if len(encoded) > MAX_B64_CHARS:
        return None, None
    # Defense in depth: verify the bytes decode as an image when Pillow exists.
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
    except ImportError:
        pass
    except Exception:
        return None, None
    _ = os.sep  # keep os import used (path handling above)
    return encoded, mime
