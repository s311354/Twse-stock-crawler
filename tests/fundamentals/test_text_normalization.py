import pytest

from domain.text_normalization import normalize_cjk_text


def test_normalize_cjk_text_removes_cjk_layout_whitespace() -> None:
    raw = (
        "部分非重要\u3000子公司\n"
        "未經會計師核\u00a0閱&amp;查 核；ＡＢＣ"
    )

    normalized = normalize_cjk_text(raw)

    assert normalized == "部分非重要子公司未經會計師核閱&查核;ABC"
    assert normalize_cjk_text(normalized) == normalized


def test_normalize_cjk_text_preserves_english_word_boundaries() -> None:
    raw = "basis  for qualified\u200b opinion &lt;reviewed&gt;"

    assert (
        normalize_cjk_text(raw)
        == "basis for qualified opinion <reviewed>"
    )


def test_normalize_cjk_text_handles_empty_values() -> None:
    assert normalize_cjk_text(None) == ""
    assert normalize_cjk_text("") == ""


@pytest.mark.parametrize(
    "separator",
    ["\n", "\r\n", "\t", "\u00a0", "\u1680", "\u180e", "\u2007",
     "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\u200b",
     "\u200c", "\u200d", "\u2060", "\ufeff"],
)
def test_normalize_cjk_text_handles_layout_separators(separator: str) -> None:
    assert normalize_cjk_text(f"非重要{separator}子公司") == "非重要子公司"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("非重要&amp;nbsp;子公司", "非重要子公司"),
        ("非重要&#x202f;子公司", "非重要子公司"),
        ("非重要＆ｎｂｓｐ；子公司", "非重要子公司"),
        ("非重要&nb\u200bsp;子公司", "非重要子公司"),
        ("ＡＢＣ（ＴＷＤ：１，２３４．５０）；２５％", "ABC(TWD:1,234.50);25%"),
        ("非重要 子公司 ABC\u00a0Ltd. 核 閱", "非重要子公司 ABC Ltd. 核閱"),
        ("未知&madeup;符號", "未知&madeup;符號"),
    ],
)
def test_normalize_cjk_text_is_idempotent_for_entities_and_width(
    raw: str, expected: str,
) -> None:
    normalized = normalize_cjk_text(raw)

    assert normalized == expected
    assert normalize_cjk_text(normalized) == normalized
