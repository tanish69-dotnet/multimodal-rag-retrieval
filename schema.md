# Multimodal RAG Data Contract (schema.md)

## 1. Input Contract: Person 1 (Parser) -> Person 2 (Retrieval)
Person 1 stores parsed research papers as JSON/dict objects adhering to this schema:

```json
{
  "paper_id": "arxiv_2308.12345",
  "title": "Attention Is All You Need",
  "elements": [
    {
      "element_id": "elem_001",
      "type": "text",
      "section": "3.1 Encoder and Decoder Stacks",
      "page": 3,
      "content": "The encoder is composed of a stack of N = 6 identical layers..."
    },
    {
      "element_id": "elem_002",
      "type": "table",
      "section": "5. Evaluation",
      "page": 7,
      "content": "| Model | BLEU (EN-DE) | BLEU (EN-FR) |\n|---|---|---|\n| Transformer (base) | 27.3 | 38.1 |",
      "caption": "Table 1: Maximum BLEU scores on translation tasks.",
      "mentions": [
        "As shown in Table 1, our base model beats previous ensemble models."
      ],
      "table_data": {
        "headers": ["Model", "BLEU (EN-DE)", "BLEU (EN-FR)"],
        "rows": [["Transformer (base)", "27.3", "38.1"]],
        "caption": "Table 1: Maximum BLEU scores on translation tasks."
      },
      "source_url": "https://arxiv.org/html/2308.12345v1",
      "source_location": "Table 1"
    },
    {
      "element_id": "elem_003",
      "type": "figure",
      "section": "3. Model Architecture",
      "page": 3,
      "content": "",
      "caption": "Figure 1: The Transformer model architecture.",
      "mentions": [
        "Figure 1 illustrates the multi-head self-attention mechanisms."
      ],
      "image_path": "data/extracted/arxiv_2308.12345/fig1.png"
    }
  ]
}
```

## 2. Output Contract: Person 2 (Retrieval) -> Person 3 (Reasoning)
Person 2 provides a ranked list of EvidenceItem objects to Person 3:

```json
[
  {
    "rank": 1,
    "score": 0.884,
    "paper_id": "arxiv_2308.12345",
    "element_id": "elem_002",
    "type": "table",
    "page": 7,
    "section": "5. Evaluation",
    "content": "| Model | BLEU (EN-DE) | BLEU (EN-FR) |\n|---|---|---|\n| Transformer (base) | 27.3 | 38.1 |",
    "caption": "Table 1: Maximum BLEU scores on translation tasks.",
    "context": "As shown in Table 1, our base model beats previous ensemble models.",
    "image_path": null,
    "table_data": {
      "headers": ["Model", "BLEU (EN-DE)", "BLEU (EN-FR)"],
      "rows": [["Transformer (base)", "27.3", "38.1"]],
      "caption": "Table 1: Maximum BLEU scores on translation tasks."
    },
    "metadata": {
      "source_url": "https://arxiv.org/html/2308.12345v1",
      "source_location": "Table 1"
    }
  }
]
```
