"""Phase 9H tests for source parsing and structured table preservation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.restore_target_paper import _parse_source


SOURCE_FIXTURE = """
<html><body>
<h1 class="ltx_title_document">Knowledge Enhanced Hybrid Neural Network for Text Matching</h1>
<div class="ltx_abstract"><h6>Abstract</h6><p class="ltx_p">We propose KEHNN for text matching.</p></div>
<section class="ltx_section"><h2 class="ltx_title_section">Introduction</h2>
  <p class="ltx_p">The model uses prior knowledge to improve matching.</p>
</section>
<section class="ltx_section"><h2 class="ltx_title_section">Experiments</h2>
  <section class="ltx_subsection"><h3 class="ltx_title_subsection">Response Selection</h3>
    <figure class="ltx_table"><figcaption>Table 4: Evaluation results on response selection</figcaption>
      <table><tr><td>R2@1</td><td>R10@1</td></tr>
        <tr><td>MLP</td><td>0.651</td></tr>
        <tr><td>KEHNN</td><td>0.786</td></tr>
      </table>
    </figure>
  </section>
</section>
</body></html>
"""


def test_phase9h_parser_preserves_source_text_and_table_values() -> None:
    title, text_elements, table = _parse_source(SOURCE_FIXTURE)

    assert title == "Knowledge Enhanced Hybrid Neural Network for Text Matching"
    assert any(element["element_id"] == "text_abstract" for element in text_elements)
    assert any("prior knowledge" in element["content"] for element in text_elements)
    assert table["element_id"] == "elem_004"
    assert table["table_data"]["headers"] == ["R2@1", "R10@1"]
    assert table["table_data"]["rows"][-1] == ["KEHNN", "0.786"]
    assert table["source_location"].endswith("Table 4")
