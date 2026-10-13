# Retrieval service — deployment notes (free-first)

Deploy target: **Hugging Face Spaces, Docker SDK, CPU Basic (2 vCPU / 16 GB,
free)**. See `../DEPLOY.md` (main repo) for the full topology and limits.

- `Dockerfile` reads `$PORT` (HF injects 7860; local default 8000).
- `README.md` front-matter sets `sdk: docker`, `app_port: 7860`.
- `render.yaml` is a paid-only warning stub — Render free (512 MB, no disk)
  cannot host this stack.
- The demo corpus (`data/processed_papers.json` + KEHNN figure artifacts)
  ships in the image and re-indexes on boot, so the public demo is always
  queryable even though **user uploads are ephemeral** (Space disk is not
  persistent; everything in `chroma_db/`, `data/ingested/`, `/tmp/p2_ingest`
  is lost on restart — warn users in the UI).
