import pandas as pd
import pandas.testing as pdt
import pytest

from domain.pipeline import DEFAULT_PIPELINE_STAGE_NAMES
from domain.pipeline import PandasPipeline
from domain.pipeline import PipelineStage


def test_pandas_pipeline_runs_raw_clean_feature_signal_report_flow() -> None:
    pipeline = PandasPipeline.from_transforms(
        clean=lambda data: data.assign(Close=pd.to_numeric(data["Close"])),
        feature_engineering=lambda data: data.assign(MOM=data["Close"].pct_change().fillna(0)),
        signal=lambda data: data.assign(Entry_Signal=data["MOM"] > 0),
        report=lambda data: data[["Close", "MOM", "Entry_Signal"]],
    )
    raw = pd.DataFrame({"Close": ["100", "105"]})

    result = pipeline.run(raw)

    assert pipeline.stage_names == DEFAULT_PIPELINE_STAGE_NAMES
    pdt.assert_frame_equal(
        result,
        pd.DataFrame({"Close": [100, 105], "MOM": [0.0, 0.05], "Entry_Signal": [False, True]}),
    )
    assert raw["Close"].tolist() == ["100", "105"]


def test_pandas_pipeline_rejects_non_dataframe_stage_output() -> None:
    pipeline = PandasPipeline([PipelineStage("bad", lambda data: ["not", "a", "frame"])])

    with pytest.raises(TypeError, match="bad"):
        pipeline.run(pd.DataFrame({"Close": [100]}))
