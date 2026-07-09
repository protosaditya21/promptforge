from promptforge.tokens import count_tokens, fits_budget


def test_empty_string_is_zero_tokens():
    assert count_tokens("") == 0


def test_count_tokens_positive_for_text():
    assert count_tokens("hello world, this is a test sentence.") > 0


def test_longer_text_has_more_tokens():
    short = "hello world"
    long = "hello world " * 50
    assert count_tokens(long) > count_tokens(short)


def test_fits_budget_true_for_small_text():
    assert fits_budget("hi", budget=100) is True


def test_fits_budget_false_for_large_text():
    big_text = "word " * 5000
    assert fits_budget(big_text, budget=10) is False
