import pandas as pd
import pytest

from trading_lab.data.base import DataValidationError, validate_ohlcv


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3], "Low": [0.5, 1.5], "Close": [1.5, 2.5], "Volume": [1, 2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )


def test_ohlcv_requires_columns() -> None:
    with pytest.raises(DataValidationError, match="Volume"):
        validate_ohlcv(frame().drop(columns="Volume"))


def test_unsorted_and_duplicate_data_are_normalized_explicitly() -> None:
    raw = pd.concat([frame().iloc[[1]], frame().iloc[[0]], frame().iloc[[1]]])
    normalized = validate_ohlcv(raw, normalize_index=True)
    assert normalized.index.is_monotonic_increasing
    assert not normalized.index.has_duplicates
    with pytest.raises(DataValidationError, match="sorted"):
        validate_ohlcv(raw)


def test_ohlcv_high_validation_reports_invalid_rows() -> None:
    raw = frame()
    raw.loc[raw.index[0], "High"] = 1

    with pytest.raises(DataValidationError, match="First invalid rows") as error:
        validate_ohlcv(raw)

    message = str(error.value)
    assert "2024-01-01" in message
    assert "Open" in message and "High" in message and "Low" in message and "Close" in message
