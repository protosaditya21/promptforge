import pytest

from promptforge.retry import RetryExhaustedError, with_retry


def test_succeeds_without_retry():
    calls = []

    @with_retry(max_attempts=3, sleep_fn=lambda _: None)
    def always_ok():
        calls.append(1)
        return "ok"

    assert always_ok() == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    calls = []

    @with_retry(max_attempts=3, base_delay=0.01, sleep_fn=lambda _: None)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3


def test_exhausts_retries_and_raises():
    @with_retry(max_attempts=2, base_delay=0.01, sleep_fn=lambda _: None)
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(RetryExhaustedError) as exc_info:
        always_fails()
    assert exc_info.value.attempts == 2


def test_only_retries_specified_exceptions():
    @with_retry(max_attempts=3, exceptions=(TimeoutError,), sleep_fn=lambda _: None)
    def raises_value_error():
        raise ValueError("should not be retried")

    with pytest.raises(ValueError):
        raises_value_error()
