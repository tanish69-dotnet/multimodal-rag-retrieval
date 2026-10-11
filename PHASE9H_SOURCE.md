# Phase 9H source and restore procedure

The local SPIQA test-split export contains the target paper's captions and QA
annotations, but not article text, table cells, pages, or image files. The
missing paper material is restored from the versioned arXiv HTML artifact:

- Paper: `1611.04684v1`
- Title: `Knowledge Enhanced Hybrid Neural Network for Text Matching`
- Source: <https://arxiv.org/html/1611.04684v1>
- Source identity: arXiv version `v1`, submitted 15 Nov 2016
- License/identity page: <https://arxiv.org/abs/1611.04684v1>

Run from this project root to fetch, hash, parse, and merge the target paper:

```text
python -m src.restore_target_paper --refresh
```

The command stores the downloaded HTML and SHA-256 manifest under
`data/source_artifacts/` and updates only the `1611.04684v1` record in
`data/processed_papers.json`. The separate repository tracks the source
artifact, manifest, and restored corpus snapshots; Chroma files remain local
because the repository intentionally ignores `chroma_db/`.

The parser extracts the real abstract and paper sections as `text` elements,
and extracts Table 4's HTML header and rows into both Markdown `content` and
optional `table_data`. Page numbers and image paths remain null because the
HTML source does not provide them; `source_url` and `source_location` preserve
the available provenance.
