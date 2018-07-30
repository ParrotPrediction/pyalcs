import pytest

from lcs import TypedList


class TestTypedList:

    def test_should_initialize_empty_int_list(self):
        assert len(TypedList(int)) is 0

    def test_should_prepopulate_list(self):
        # given
        oktypes = (int,)
        elems = [1, 2, 3]

        # when
        lst = TypedList(oktypes, *elems)

        # then
        assert len(lst) == 3
        for el in elems:
            assert el in lst

    def test_should_fail_when_prepopulating_list(self):
        # given
        oktypes = (int,)
        elems = [1, 2, "3"]

        # when
        with pytest.raises(TypeError) as _:
            TypedList(oktypes, *elems)

    def test_should_getitem(self):
        # given
        oktypes = (int,)
        elems = [1, 2, 3]

        # when
        lst = TypedList(oktypes, *elems)

        # then
        assert lst[0] == 1
        assert lst[1] == 2
        assert lst[2] == 3

    def test_should_insert_items(self):
        # given
        oktypes = (int,)
        lst = TypedList(oktypes)

        # when
        lst.insert(0, 1)
        lst.insert(1, 2)

        # then
        assert len(lst) == 2

    def test_should_fail_when_inserting_items(self):
        # given
        oktypes = (int,)
        lst = TypedList(oktypes)

        # then
        with pytest.raises(TypeError) as _:
            lst.insert(0, "1")

    def test_should_append_items(self):
        # given
        lst = TypedList((int,), *[1, 2, 3])

        # when
        lst.append(4)

        # then
        assert len(lst) == 4
        assert 4 in lst

    def test_should_fail_when_appending(self):
        # given
        lst = TypedList((int,), *[1, 2, 3])

        # then
        with pytest.raises(TypeError) as _:
            lst.append("4")

    def test_should_delete_item(self):
        # given
        lst = TypedList((int,), *[1, 2, 3])

        # when
        del lst[0]

        # then
        assert len(lst) == 2
        assert 1 not in lst
