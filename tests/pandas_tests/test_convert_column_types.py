import numpy as np
import pandas as pd

from workbench.core.transforms.pandas_transforms import PandasToFeatures

convert = PandasToFeatures.convert_column_types


class TestBoolConversion:
    """Bool / nullable boolean -> Integral (int32), or Fractional (float64) when NAs are present."""

    def test_numpy_bool_to_int32(self):
        df = pd.DataFrame({"flag": pd.array([True, False, True], dtype="bool")})
        out = convert(df)
        assert out["flag"].dtype == np.int32
        assert out["flag"].tolist() == [1, 0, 1]

    def test_nullable_boolean_without_na_to_int32(self):
        df = pd.DataFrame({"flag": pd.array([True, False, True], dtype="boolean")})
        out = convert(df)
        assert out["flag"].dtype == np.int32
        assert out["flag"].tolist() == [1, 0, 1]

    def test_nullable_boolean_with_na_to_float64(self):
        """Regression: previously raised 'cannot convert NA to integer'."""
        df = pd.DataFrame({"flag": pd.array([True, pd.NA, False], dtype="boolean")})
        out = convert(df)
        assert out["flag"].dtype == np.float64
        assert out["flag"].iloc[0] == 1.0
        assert np.isnan(out["flag"].iloc[1])
        assert out["flag"].iloc[2] == 0.0


class TestCategoryConversion:
    """Category -> String (object dtype)."""

    def test_category_to_str(self):
        df = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
        out = convert(df)
        assert out["c"].dtype == object
        assert out["c"].tolist() == ["a", "b", "a"]


class TestDatetimeConversion:
    """Datetime -> ISO-8601 string."""

    def test_datetime_to_iso8601_string(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02 03:04:05", "2024-06-07 08:09:10"])})
        out = convert(df)
        assert str(out["ts"].dtype) == "string"
        assert out["ts"].iloc[0].startswith("2024-01-02T03:04:05")
        assert out["ts"].iloc[1].startswith("2024-06-07T08:09:10")


class TestNullableNumericDowncast:
    """Pandas nullable Int/Float -> numpy equivalents; Int with NAs falls back to float64."""

    def test_nullable_int64_without_na_to_numpy_int64(self):
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="Int64")})
        out = convert(df)
        assert out["x"].dtype == np.int64
        assert out["x"].tolist() == [1, 2, 3]

    def test_nullable_int64_with_na_to_float64(self):
        df = pd.DataFrame({"x": pd.array([1, pd.NA, 3], dtype="Int64")})
        out = convert(df)
        assert out["x"].dtype == np.float64
        assert out["x"].iloc[0] == 1.0
        assert np.isnan(out["x"].iloc[1])

    def test_nullable_uint32_without_na_to_numpy_uint32(self):
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="UInt32")})
        out = convert(df)
        assert out["x"].dtype == np.uint32

    def test_nullable_float64_with_na_to_numpy_float64(self):
        df = pd.DataFrame({"x": pd.array([1.5, pd.NA, 3.5], dtype="Float64")})
        out = convert(df)
        assert out["x"].dtype == np.float64
        assert out["x"].iloc[0] == 1.5
        assert np.isnan(out["x"].iloc[1])

    def test_numpy_dtypes_left_alone(self):
        df = pd.DataFrame(
            {
                "i": np.array([1, 2, 3], dtype=np.int64),
                "f": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            }
        )
        out = convert(df)
        assert out["i"].dtype == np.int64
        assert out["f"].dtype == np.float64


class TestMixedDataFrame:
    """Realistic mixed-type frame — verify all conversions happen in one pass without cross-talk."""

    def test_mixed_columns_all_converted(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "flag": pd.array([True, pd.NA, False], dtype="boolean"),
                "cat": pd.Categorical(["x", "y", "x"]),
                "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "nullable_int": pd.array([10, pd.NA, 30], dtype="Int64"),
                "plain_float": [1.1, 2.2, 3.3],
            }
        )
        out = convert(df)
        assert out["flag"].dtype == np.float64
        assert out["cat"].dtype == object
        assert str(out["ts"].dtype) == "string"
        assert out["nullable_int"].dtype == np.float64
        assert out["plain_float"].dtype == np.float64
