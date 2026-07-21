import numpy as np
import pandas as pd
import pytest

from trading_lab.data.base import DataValidationError, validate_ohlcv


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [1, 2],
        },
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
    assert "High - max(Open, Close, Low)" in message
    assert "Low - min(Open, Close, High)" in message


def test_ohlcv_accepts_high_one_floating_point_step_below_close() -> None:
    raw = frame()
    raw.loc[raw.index[0], "High"] = float(np.nextafter(raw.loc[raw.index[0], "Close"], 0.0))

    validated = validate_ohlcv(raw)

    assert validated.loc[raw.index[0], "High"] < validated.loc[raw.index[0], "Close"]


def test_ohlcv_accepts_low_one_floating_point_step_above_open() -> None:
    raw = frame()
    raw.loc[raw.index[0], "Low"] = float(np.nextafter(raw.loc[raw.index[0], "Open"], float("inf")))

    validated = validate_ohlcv(raw)

    assert validated.loc[raw.index[0], "Low"] > validated.loc[raw.index[0], "Open"]


def test_ohlcv_rejects_significantly_low_high() -> None:
    raw = frame()
    raw.loc[raw.index[0], "High"] = raw.loc[raw.index[0], "Close"] - 0.01

    with pytest.raises(DataValidationError, match="High must be at least"):
        validate_ohlcv(raw)


def test_ohlcv_rejects_significantly_high_low() -> None:
    raw = frame()
    raw.loc[raw.index[0], "Low"] = raw.loc[raw.index[0], "Open"] + 0.01

    with pytest.raises(DataValidationError, match="Low must be at most"):
        validate_ohlcv(raw)


def test_ohlcv_diagnostic_uses_high_precision_values_for_real_error() -> None:
    raw = frame()
    raw.loc[raw.index[0], "High"] = 1.2345678901234567
    raw.loc[raw.index[0], "Close"] = 1.9876543210987654

    with pytest.raises(DataValidationError) as error:
        validate_ohlcv(raw)

    message = str(error.value)
    assert "1.2345678901234567" in message
    assert "1.9876543210987654" in message
    assert "-0.7530864309753087" in message
