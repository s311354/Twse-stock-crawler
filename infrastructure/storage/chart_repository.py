"""Chart rendering infrastructure for TWSE reports."""

from __future__ import annotations

from pathlib import Path

from config.settings import CHART_PATH


def save_profit_ratio_chart(
    duration: int,
    maxprofitratios: list[list[float]],
    stocknumbers: list[str],
    stocksprice: list[float | None],
    output_path: str | Path = CHART_PATH,
) -> Path:
    """Render the max-profit-ratio trend chart.

    Matplotlib is imported lazily so tests that only import the CLI do not pay
    the font-cache startup cost unless chart rendering is used.
    """

    import matplotlib.pyplot as plt

    x = [index for index in range(duration)]
    fig, ax = plt.subplots()

    for item, ratios in enumerate(maxprofitratios):
        ax.plot(x, ratios, label="Stock {} ; Opening price {}".format(stocknumbers[item], stocksprice[item]))

    ax.set_xlabel("Period")
    ax.set_ylabel("Max Profit Ratio")
    ax.set_title("The Trend of Performance Indicators for Taiwan Stock Market ($NTD)")
    ax.legend()

    output = Path(output_path)
    fig.savefig(output)
    plt.close(fig)
    return output

