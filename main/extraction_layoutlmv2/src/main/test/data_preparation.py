import pytest
import src.main.extraction.training_utility as tu

@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Test case 1: Happy path, y1 values are within 15, expect difference of x1 values
        ({"x1": 10, "y1": 20}, {"x1": 5, "y1": 25}, 5, "happy_path_within_15"),

        # Test case 2: Happy path, y1 values are not within 15, expect difference of y1 values
        ({"x1": 10, "y1": 20}, {"x1": 5, "y1": 40}, -20, "happy_path_not_within_15"),

        # Test case 3: Edge case, y1 values are exactly 15 apart, expect difference of x1 values
        ({"x1": 10, "y1": 20}, {"x1": 5, "y1": 35}, 5, "edge_case_exactly_15"),

        # Test case 4: Error case, a or b is not a dictionary
        (10, {"x1": 5, "y1": 25}, TypeError, "error_case_not_dictionary"),

        # Test case 5: Error case, a or b does not contain 'x1' or 'y1'
        ({"x1": 10}, {"x1": 5, "y1": 25}, KeyError, "error_case_missing_key"),
    ],
)
def test_contour_sort(a, b, expected, test_case):
    if "error_case" in test_case:
        with pytest.raises(expected):
            # Act
            tu.contour_sort(a, b)
    else:
        # Act
        result = tu.contour_sort(a, b)

        # Assert
        assert result == expected