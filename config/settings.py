"""Central settings for filesystem paths and runtime defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHART_PATH = PROJECT_ROOT / "max_profit_analysis_chart.png"
ENV_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example")
QUERY_WAIT_SECONDS = 12
