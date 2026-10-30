"""No network or real account access in benchmark orchestration tests."""
import socket

import pytest

from examples.benchmark_daily_full_scope import compare_measurements, offline_only


def test_comparison_requires_every_complete_business_identity():
    cold = dict(input_id="input", decisions="same", reports="same", account="same", state="same")
    for key in ("decisions", "reports", "account", "state"):
        warm = dict(cold, **{key: "changed"})
        with pytest.raises(ValueError, match="COMPLETE_PARITY_FAILED:" + key):
            compare_measurements([cold, warm], ("decisions", "reports", "account", "state"))
    compare_measurements([cold, dict(cold)], ("decisions", "reports", "account", "state"))


def test_missing_identity_or_different_fixed_input_cannot_pass():
    with pytest.raises(ValueError, match="COMPLETE_PARITY_FAILED"):
        compare_measurements([dict(input_id="a"), dict(input_id="a")], ("reports",))
    with pytest.raises(ValueError, match="INPUT_ID_CHANGED"):
        compare_measurements([dict(input_id="a", reports="x"), dict(input_id="b", reports="x")], ("reports",))
    with pytest.raises(ValueError, match="REQUIRES_COLD_AND_WARM"):
        compare_measurements([dict(input_id="a")], ())


def test_offline_guard_denies_http_smtp_and_restores_socket_apis():
    original = socket.create_connection, socket.socket.connect, socket.socket.connect_ex
    with offline_only():
        with pytest.raises(RuntimeError, match="NETWORK_FORBIDDEN"):
            socket.create_connection(("example.invalid", 443))
        with socket.socket() as handle:
            with pytest.raises(RuntimeError, match="NETWORK_FORBIDDEN"):
                handle.connect(("127.0.0.1", 25))
            with pytest.raises(RuntimeError, match="NETWORK_FORBIDDEN"):
                handle.connect_ex(("127.0.0.1", 25))
    assert (socket.create_connection, socket.socket.connect, socket.socket.connect_ex) == original
