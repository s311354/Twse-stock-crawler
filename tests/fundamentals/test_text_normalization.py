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
