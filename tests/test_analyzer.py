import pytest
import pandas as pd
from app.analyzer import DataAnalyzer

class TestAnalyzer:
    """Data analyzer tests"""

    def test_aggregate_data_sum(self):
        """Test data aggregation with sum"""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40]
        })

        result = DataAnalyzer.aggregate_data(df, "category", "value", "sum")
        assert result["A"] == 40
        assert result["B"] == 60

    def test_aggregate_data_mean(self):
        """Test data aggregation with mean"""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40]
        })

        result = DataAnalyzer.aggregate_data(df, "category", "value", "mean")
        assert result["A"] == 20.0
        assert result["B"] == 30.0

    def test_calculate_statistics(self):
        """Test statistics calculation"""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        stats = DataAnalyzer.calculate_statistics(df, "value")

        assert stats["mean"] == 3.0
        assert stats["sum"] == 15
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5

    def test_filter_dataframe(self):
        """Test DataFrame filtering"""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40]
        })

        filtered = DataAnalyzer.filter_dataframe(df, {"category": "A"})
        assert len(filtered) == 2
        assert all(filtered["category"] == "A")
