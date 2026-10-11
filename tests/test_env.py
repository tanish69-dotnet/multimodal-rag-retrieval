"""Environment verification script for the multimodal RAG retrieval pipeline.

Checks that all required packages are importable and that the embedding
pipeline (BAAI/bge-small-en-v1.5 via sentence-transformers) works end to end.
"""

import sys

FAILURES = []


def check_import(module_name: str, package_name: str | None = None) -> None:
    """Try importing `module_name` and record whether it succeeded."""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] import {module_name} (version: {version})")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        print(f"[FAIL] import {module_name}: {exc}")
        FAILURES.append((package_name or module_name, exc))


def normalize(vector):
    """L2-normalize a vector in place-safe way and return it."""
    import numpy as np

    vector = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def main() -> int:
    print(f"Python {sys.version.split()[0]} ({sys.executable})\n")

    # 1. Required library imports.
    check_import("chromadb")
    check_import("rank_bm25")
    check_import("sentence_transformers")
    check_import("datasets")
    check_import("pandas")
    check_import("numpy")  # needed for normalization

    if FAILURES:
        print("\nMissing packages:")
        for name, _ in FAILURES:
            print(f"  - {name}")
        print("\nTry: pip install -r requirements.txt")
        return 1

    # 2. Embedding pipeline smoke test with BGE-small.
    print("\nLoading BAAI/bge-small-en-v1.5 via sentence-transformers ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    test_text = "Testing embedding pipeline"
    embedding = model.encode(test_text, normalize_embeddings=False)
    embedding = normalize(embedding)

    print(f"Encoded text       : {test_text!r}")
    print(f"Embedding shape    : {embedding.shape}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Vector norm        : {float((embedding ** 2).sum() ** 0.5):.6f}")
    print("\nEnvironment verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())