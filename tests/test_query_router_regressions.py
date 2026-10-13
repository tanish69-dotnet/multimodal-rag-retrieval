"""Focused retrieval intent regressions that do not initialize model indexes."""

from src.query_router import classify_intent


def test_paraphrased_best_method_is_table_intent():
    assert classify_intent("Which method performs best?") == "table"


def test_explicit_figure_reference_wins_over_generic_text_words():
    assert classify_intent("Explain the architecture in Figure 1.") == "figure"


def test_explicit_figure_reference_wins_over_comparative_words():
    # "best" appears, but the question targets Figure 1 — not a table.
    assert classify_intent("Which component is shown best in Figure 1?") == "figure"


def test_explicit_table_reference_still_routes_table():
    assert classify_intent("What does Table 1 show?") == "table"


def test_architecture_without_reference_routes_figure():
    assert classify_intent("Describe the architecture diagram.") == "figure"


def test_bare_number_question_is_not_table_intent():
    assert classify_intent("What year was the paper published?") == "general"
