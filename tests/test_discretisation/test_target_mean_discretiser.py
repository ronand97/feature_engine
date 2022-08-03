import pandas as pd
import pytest

from feature_engine.discretisation import MeanDiscretiser


def test_equal_frequency_strategy():
    data = {
        "var_A": list(range(1, 11)),
        "var_B": list(range(2, 22, 2)),
        "var_C": ["A"] * 3 + ["B"] + ["C"] * 4 + ["D"] * 2,
        "var_D": list(range(3, 33, 3)),
    }
    df = pd.DataFrame(data)
    target = list(range(10))

    transformer = MeanDiscretiser(
        variables=["var_A", "var_D"],
        bins=2,
        strategy="equal_frequency",
    )

    df_tr = transformer.fit_transform(df, target)

    expected_results = {
        "var_A": [2.0] * 5 + [7.0] * 5,
        "var_B": list(range(2, 22, 2)),
        "var_C": ["A"] * 3 + ["B"] + ["C"] * 4 + ["D"] * 2,
        "var_D": [2.0] * 5 + [7.0] * 5,
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert df_tr.equals(expected_results_df)


def test_equal_width_strategy():
    data = {
        "var_W": list(range(5, 55, 5)),
        "var_X": ["W"] * 3 + ["X"] + ["Y"] * 4 + ["Z"] * 2,
        "var_Y": list(range(3, 33, 3)),
        "var_Z": list(range(4, 44, 4)),
    }
    df = pd.DataFrame(data)
    target = list(range(10, 30, 2))
    transformer = MeanDiscretiser(
        variables=["var_Y", "var_Z"],
        bins=3,
        strategy="equal_width",
    )
    df_tr = transformer.fit_transform(df, target)

    expected_results = {
        "var_W": list(range(5, 55, 5)),
        "var_X": ["W"] * 3 + ["X"] + ["Y"] * 4 + ["Z"] * 2,
        "var_Y": [13.0] * 4 + [20.0] * 3 + [26.0] * 3,
        "var_Z": [13.0] * 4 + [20.0] * 3 + [26.0] * 3,
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert df_tr.equals(expected_results_df)


@pytest.mark.parametrize("bins_value", ["other", 0.5, [1]])
def test_error_when_bins_not_number(bins_value):
    with pytest.raises(ValueError):
        MeanDiscretiser(bins=bins_value)


@pytest.mark.parametrize("strategy_value", ["other", 0.5, [1]])
def test_error_when_strategy_not_valid(strategy_value):
    with pytest.raises(ValueError):
        MeanDiscretiser(bins=strategy_value)
