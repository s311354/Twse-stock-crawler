from domain.security_identity import SecurityType
from domain.security_identity import resolve_security_identity


def test_common_stock_keeps_the_same_market_and_issuer_symbol() -> None:
    identity = resolve_security_identity("2330", "台積電")

    assert identity.market_symbol == "2330"
    assert identity.issuer_symbol == "2330"
    assert identity.security_type is SecurityType.COMMON
    assert identity.low_entry_model_supported is True


def test_preferred_stock_maps_to_issuer_without_losing_market_identity() -> None:
    identity = resolve_security_identity("3036a", "文曄甲特")

    assert identity.market_symbol == "3036A"
    assert identity.issuer_symbol == "3036"
    assert identity.security_type is SecurityType.PREFERRED
    assert identity.low_entry_model_supported is False


def test_unknown_alpha_suffix_is_not_blindly_stripped() -> None:
    identity = resolve_security_identity("1234A", "測試證券")

    assert identity.market_symbol == "1234A"
    assert identity.issuer_symbol == "1234A"
    assert identity.security_type is SecurityType.OTHER
