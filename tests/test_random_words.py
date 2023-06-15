import pytest
from src.feature_salad.utils.random_words import RandomWords


@pytest.fixture
def rw() -> RandomWords:
    rw = RandomWords()
    return rw

def test_return_n_words(rw) -> None:
    words = rw.get_words(3)
    assert len(words) == 3, "returning wrong number of words."
    assert isinstance(words[0], str), "not returning words."