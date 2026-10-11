"""Phase 3 verification: baseline dense retrieval end to end.

1. Chunker spec check on a synthetic text element (the current SPIQA subset has
   no text elements, so full-text behavior is verified explicitly):
     - ~400-word chunks with ~50-word overlap
     - "References"-style sections skipped
     - contextual header prepended, metadata propagated
2. Index ``data/processed_papers.json`` (figure/table captions) into the
   persistent ``chroma_db/academic_rag_dense`` collection.
3. Retrieve top-3 EvidenceItems for 5 questions from
   ``data/evaluation_questions.json`` and print score / page / section / content.

Run from the project root:  python tests/test_dense_retrieval.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    chunk_text_element,
    is_reference_section,
)
from src.data_loader import load_local_subset
from src.dense_retriever import DenseRetriever

TOP_K = 3
NUM_QUESTIONS = 5

try:  # Windows consoles default to cp1252; keep unicode printable
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def tx(value) -> str:
    """Console-safe string (never raises UnicodeEncodeError)."""
    text = "" if value is None else str(value)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii")


def fmt(value, width):
    text = tx(value)
    if len(text) > width - 3:
        text = text[: width - 3] + "..."
    return text.ljust(width)


def check_text_chunker() -> None:
    """Exercise the section-aware chunker with a synthetic schema text element."""
    print("=" * 78)
    print("1) CHUNKER SPEC CHECK (synthetic text element)")
    print("=" * 78)

    sentence = ("We compare our model against strong baselines on several benchmark "
                "datasets and report the mean and standard deviation over five seeds. ")
    content = sentence * 60  # ~660 words -> 2+ overlapping 400-word chunks

    paper_title = "Example Academic Paper"
    element = {
        "paper_id": "synth_paper_001",
        "element_id": "elem_text_001",
        "type": "text",
        "section": "3. Experiments",
        "page": 3,
        "content": content,
    }

    chunks = chunk_text_element(element, title=paper_title)
    assert chunks, "chunker returned no chunks for a long text element"
    print(f"chunks produced        : {len(chunks)} (chunk_size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_OVERLAP})")
    for i, chunk in enumerate(chunks[:3]):
        words = len(chunk["raw_text"].split())
        print(f"  c{i:02d}: ~{words:<4} words | header={chunk['header']!r}")
        if i == 0:
            assert chunk["content"].startswith(f"Paper: {paper_title} | Section: 3. Experiments"), "contextual header missing"
            assert chunk["page"] == 3 and chunk["section"] == "3. Experiments", "metadata not propagated"
            assert chunk["paper_id"] == "synth_paper_001" and chunk["element_id"] == "elem_text_001"
            assert chunk["type"] == "text"

    # Overlap check between chunk 0 and chunk 1.
    if len(chunks) >= 2:
        tail0 = " ".join(chunks[0]["raw_text"].split()[-DEFAULT_OVERLAP:])
        head1 = " ".join(chunks[1]["raw_text"].split()[:DEFAULT_OVERLAP])
        overlap_ok = tail0 == head1
        print(f"overlap boundary      : {'OK' if overlap_ok else 'MISMATCH'} "
              f"({tail0[:40]!r} ... {head1[:40]!r})")
        assert overlap_ok, "sliding-window overlap mismatch"

    # Reference-section skip.
    for heading in ["References", "13. References", "Bibliography", "Acknowledgements"]:
        skipped = chunk_text_element({**element, "section": heading}, title=paper_title)
        assert not skipped, f"{heading!r} should have been skipped"
    print("reference sections    : References / Bibliography / Acknowledgements skipped OK")
    print("header + metadata     : OK (Paper & Section header prepended; paper_id/element_id/type/page/section kept)")


def main() -> int:
    check_text_chunker()

    print("\n" + "=" * 78)
    print("2) LOAD LOCAL SUBSET")
    print("=" * 78)
    papers, questions = load_local_subset()
    n_text = sum(1 for p in papers for e in p["elements"] if (e.get("type") or "") == "text")
    n_visual = sum(1 for p in papers for e in p["elements"] if (e.get("type") or "") != "text")
    print(f"papers={len(papers)}  elements: text={n_text} visual(figure/table)={n_visual}  questions={len(questions)}")
    print(f"(note: current subset has no text elements -> chunker indexes figure/table captions; "
          f"text path verified in step 1)")

    print("\n" + "=" * 78)
    print("3) INDEX PAPERS INTO ChromaDB")
    print("=" * 78)
    retriever = DenseRetriever()
    n_chunks = retriever.index_papers(papers)
    print(f"collection '{retriever.collection_name}' now holds {retriever.count()} chunks "
          f"in {retriever.chroma_dir}")
    assert n_chunks > 0, "index is empty - nothing retrieved"
    assert n_chunks == retriever.count(), "upserted count does not match collection count"

    # Papers with at least one evaluable question (for the optional filter demo).
    paper_ids = sorted({q["paper_id"] for q in questions})

    print("\n" + "=" * 78)
    print(f"4) TOP-{TOP_K} RETRIEVAL FOR {NUM_QUESTIONS} SAMPLE QUESTIONS")
    print("=" * 78)

    hits_total = 0
    gt_hits = 0
    for qi, q in enumerate(questions[:NUM_QUESTIONS]):
        gt_element = q.get("reference_element_id")
        hits = retriever.retrieve(q["question"], k=TOP_K)
        hits_total += len(hits)
        hit_element_ids = {h["element_id"] for h in hits}
        gt_ok = gt_element in hit_element_ids
        gt_hits += int(gt_ok)

        print(tx(f"\nQ{qi + 1} [{q['paper_id']}] {q['question'][:90]}"))
        print(tx(f"   ground truth: element={gt_element} ({q.get('reference_type')}) -> "
                 f"{'TOP-K HIT' if gt_ok else 'not in top-k'}"))
        print("   rank  score   page section             source_image / snippet")
        for h in hits:
            source = h.get("source_image") or h.get("chunk_id") or ""
            snippet = (h.get("context") or h.get("content") or "").replace("\n", " ")
            print(f"   {h['rank']:<4} {h['score'] if h['score'] is not None else 'n/a':>5} "
                  f"{fmt(h['page'], 4)} {fmt(h['section'] or '-', 15)} {fmt(source, 24)} {fmt(snippet, 60)}")

    print("\n" + "=" * 78)
    print("5) PAPER-LEVEL METADATA FILTER (paper_ids=...)")
    print("=" * 78)
    pid = paper_ids[0]
    filtered = retriever.retrieve(questions[0]["question"], paper_ids=[pid], k=TOP_K)
    all_same_paper = all(h["paper_id"] == pid for h in filtered)
    print(f"paper filter: {pid} -> {len(filtered)} results, all from requested paper: {all_same_paper}")
    for h in filtered:
        snippet = (h.get("context") or h.get("content") or "").replace("\n", " ")
        print(tx(f"   rank={h['rank']} score={h['score']} element={h['element_id']} "
                 f"type={h['type']} page={h['page']} section={h['section']!r} | {snippet[:60]}"))
    assert all_same_paper or len(questions[0].get("paper_id")) == 0

    print("\n" + "=" * 78)
    summary = f"retrieved={hits_total / NUM_QUESTIONS:.1f} hits/question avg, " \
              f"ground-truth in top-{TOP_K}: {gt_hits}/{NUM_QUESTIONS}"
    print(f"SUMMARY: {summary}")
    print("DENSE RETRIEVAL PIPELINE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())