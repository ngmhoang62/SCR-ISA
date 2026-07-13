import pytest
from src.evaluation import Evaluator

def test_normalize_text():
    assert Evaluator.normalize_text("  Hello   World  ") == "hello world"
    assert Evaluator.normalize_text("\nTest\n") == "test"
    assert Evaluator.normalize_text("") == ""

def test_extract_boxed_answer():
    assert Evaluator.extract_boxed_answer("The result is \\boxed{42}.") == "42"
    assert Evaluator.extract_boxed_answer("No brackets here") == "No brackets here"
    assert Evaluator.extract_boxed_answer("") == ""

def test_exact_match():
    # Test matching with boxed extraction
    res1 = Evaluator.exact_match(" \\boxed{Apple} ", "apple")
    assert res1["exact_match"] is True
    assert res1["extracted_prediction"] == "Apple"

    # Test mismatches
    res2 = Evaluator.exact_match(" \\boxed{Orange} ", "apple")
    assert res2["exact_match"] is False
    assert res2["extracted_prediction"] == "Orange"
