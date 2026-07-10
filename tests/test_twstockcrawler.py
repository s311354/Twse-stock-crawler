from twstockcrawler import TwStockCrawler


class EmptyPayloadResponse:
    ok = True
    status_code = 200

    def json(self) -> dict:
        return {
            "date": "20260515",
            "groups": [],
            "stat": "OK",
            "tables": [{"fields": ["證券代號", "證券名稱"], "data": []}],
        }


class TaiexPayloadResponse:
    ok = True
    status_code = 200

    def json(self) -> dict:
        return {
            "date": "20260709",
            "stat": "OK",
            "tables": [
                {
                    # Deliberately reordered to verify schema-based lookup.
                    "fields": ["漲跌點數", "收盤指數", "指數"],
                    "data": [
                        ["10.0", "50,000.00", "寶島股價指數"],
                        ["379.80", "45,354.61", "發行量加權股價指數"],
                    ],
                }
            ],
        }


def test_get_stocktype_data_returns_empty_list_when_twse_tables_have_no_rows(monkeypatch) -> None:
    crawler = TwStockCrawler()

    monkeypatch.setattr(crawler, "_request_stocktype_data", lambda query_params, date_time: EmptyPayloadResponse())

    assert crawler.get_stocktype_data("20260515", 13) == []


def test_get_taiex_close_uses_one_index_only_request_and_schema_lookup(monkeypatch) -> None:
    crawler = TwStockCrawler()
    calls: list[dict[str, str | int]] = []

    def request_taiex(query_params: dict[str, str | int], date_time: str) -> TaiexPayloadResponse:
        calls.append(query_params)
        assert date_time == "20260709"
        return TaiexPayloadResponse()

    monkeypatch.setattr(crawler, "_request_stocktype_data", request_taiex)

    assert crawler.get_taiex_close("20260709") == 45354.61
    assert len(calls) == 1
    assert calls[0]["type"] == "IND"
