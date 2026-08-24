import os

import pytest

from src.utils.env import Env


@pytest.fixture(autouse=True)
def _cleanup(monkeypatch):
    yield
    os.environ.pop("ENV_TEST_STR", None)


def test_get_bool_works_without_instantiation():
    os.environ["ENV_TEST_STR"] = "yes"
    assert Env.get_bool("ENV_TEST_STR", default=False) is True


def test_get_bool_missing_returns_default():
    os.environ.pop("ENV_TEST_STR", None)
    assert Env.get_bool("ENV_TEST_STR", default=True) is True


def test_get_str_strips_whitespace():
    os.environ["ENV_TEST_STR"] = "  hello  "
    assert Env.get_str("ENV_TEST_STR") == "hello"


def test_get_int_invalid_falls_back_to_default():
    os.environ["ENV_TEST_STR"] = "not-a-number"
    assert Env.get_int("ENV_TEST_STR", default=7) == 7
