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


def test_get_stocktype_data_returns_empty_list_when_twse_tables_have_no_rows(monkeypatch) -> None:
    crawler = TwStockCrawler()

    monkeypatch.setattr(crawler, "_request_stocktype_data", lambda query_params, date_time: EmptyPayloadResponse())

    assert crawler.get_stocktype_data("20260515", 13) == []
