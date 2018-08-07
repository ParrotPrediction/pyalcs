import pytest

from lcs import check_types


class TestUtils:

    def test_allow_good_types_without_exception(self):
        check_types((str,), "foo")
        check_types((int,), 2)
        check_types((int, str), 2)

    def test_deny_mismatched_types(self):
        with pytest.raises(TypeError) as _:
            check_types((str,), 5)
