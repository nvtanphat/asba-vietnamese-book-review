from __future__ import annotations

from absa_core.preprocessing.formatters import normalize_format
from absa_core.preprocessing.noise_cleaner import normalize_noise, strip_html
from absa_core.preprocessing.quality_filter import (
    drop_noise_rows,
    is_digit_only,
    is_meaningful_text,
    is_symbol_only,
    normalize_for_duplicate,
)
from absa_core.preprocessing.unicode_norm import normalize_unicode
from absa_core.preprocessing.vocab_norm import normalize_vocab


class TestNormalizeUnicode:
    def test_none_and_nan_return_none(self):
        assert normalize_unicode(None) is None

    def test_strips_control_characters(self):
        result = normalize_unicode("hello\x00\x01world")
        assert result == "helloworld"

    def test_keeps_newline_tab_and_zero_width_joiner(self):
        result = normalize_unicode("a\nb\tc‍d")
        assert "\n" in result
        assert "\t" in result
        assert "‍" in result

    def test_nfc_normalizes_combining_characters(self):
        # "e" + combining acute accent (NFD) should collapse to precomposed "é" (NFC).
        decomposed = "é"
        result = normalize_unicode(decomposed)
        assert result == "é"
        assert len(result) == 1


class TestNormalizeFormat:
    def test_none_returns_none(self):
        assert normalize_format(None) is None

    def test_collapses_repeated_punctuation_to_two(self):
        assert normalize_format("that qua!!!!!") == "that qua!!"
        assert normalize_format("that sao??????") == "that sao??"

    def test_removes_zero_width_characters(self):
        assert normalize_format("hell​o") == "hello"

    def test_collapses_whitespace_and_trims(self):
        assert normalize_format("  nhieu   khoang   trang  ") == "nhieu khoang trang"


class TestNoiseCleaner:
    def test_strip_html_removes_tags(self):
        assert strip_html("<b>hay</b> qua") == "hay qua"

    def test_strip_html_no_tags_returns_original(self):
        assert strip_html("khong co the nao") == "khong co the nao"

    def test_normalize_noise_replaces_url(self):
        result = normalize_noise("xem tai https://example.com/abc nhe")
        assert "__url__" in result
        assert "example.com" not in result

    def test_normalize_noise_replaces_email(self):
        result = normalize_noise("lien he toi email@example.com giup toi")
        assert "__email__" in result
        assert "@" not in result

    def test_normalize_noise_none_returns_none(self):
        assert normalize_noise(None) is None


class TestQualityFilter:
    def test_is_digit_only(self):
        assert is_digit_only("123456") is True
        # A space still counts as non-digit content, even though it gets collapsed
        # to a single space by normalize_for_duplicate rather than removed.
        assert is_digit_only("123 456") is False
        assert is_digit_only("sach hay 123") is False

    def test_is_symbol_only(self):
        assert is_symbol_only("!!!...???") is True
        assert is_symbol_only("sach hay") is False
        assert is_symbol_only("") is False

    def test_is_meaningful_text_rejects_blank_markers(self):
        for marker in ("", "null", "none", "nan", "#name?"):
            assert is_meaningful_text(marker) is False

    def test_is_meaningful_text_rejects_short_text(self):
        assert is_meaningful_text("hay qua", min_chars=10) is False

    def test_is_meaningful_text_accepts_normal_review(self):
        assert is_meaningful_text("Sach rat hay, giao hang nhanh", min_chars=10) is True

    def test_normalize_for_duplicate_is_case_and_space_insensitive(self):
        a = normalize_for_duplicate("  Sach RAT hay  ")
        b = normalize_for_duplicate("sach rat hay")
        assert a == b

    def test_drop_noise_rows_removes_short_and_duplicate_rows(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "content": [
                    "Sach rat hay, dang doc",
                    "ok",  # too short
                    "Sach rat hay, dang doc",  # duplicate of row 0
                    "1234567890",  # digit only
                    "Giao hang nhanh, dong goi ky",
                ]
            }
        )
        result = drop_noise_rows(frame, text_column="content", min_chars=10)
        assert list(result["content"]) == [
            "Sach rat hay, dang doc",
            "Giao hang nhanh, dong goi ky",
        ]


class TestVocabNorm:
    def test_none_returns_none(self):
        assert normalize_vocab(None) is None

    def test_collapses_elongated_vowel_keeps_two_chars(self):
        assert normalize_vocab("hayyyyy") == "hayy"

    def test_collapses_elongated_consonant_at_word_end_to_one_char(self):
        assert normalize_vocab("ngonnnn") == "ngon"

    def test_preserves_punctuation_and_spacing(self):
        assert normalize_vocab("hay qua!!!") == "hay qua!!!"
