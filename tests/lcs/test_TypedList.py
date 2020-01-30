import pytest

from lcs import TypedList


class TestTypedList:

    def test_should_initialize_empty_int_list(self):
        assert len(TypedList(oktypes=int)) is 0

    def test_should_prepopulate_list(self):
        # given
        oktypes = (int,)
        elems = [1, 2, 3]

        # when
        lst = TypedList(*elems, oktypes=oktypes)

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
        lst = TypedList(*elems, oktypes=oktypes)

        # then
        assert lst[0] == 1
        assert lst[1] == 2
        assert lst[2] == 3

    def test_should_insert_items(self):
        # given
        oktypes = (int,)
        lst = TypedList(oktypes=oktypes)

        # when
        lst.insert(0, 1)
        lst.insert(1, 2)

        # then
        assert len(lst) == 2

    def test_should_fail_when_inserting_items(self):
        # given
        oktypes = (int,)
        lst = TypedList(oktypes=oktypes)

        # then
        with pytest.raises(TypeError) as _:
            lst.insert(0, "1")

    def test_should_append_items(self):
        # given
        lst = TypedList(*[1, 2, 3], oktypes=(int,))

        # when
        lst.append(4)

        # then
        assert len(lst) == 4
        assert 4 in lst

    def test_should_fail_when_appending(self):
        # given
        lst = TypedList(*[1, 2, 3], oktypes=(int,))

        # then
        with pytest.raises(TypeError) as _:
            lst.append("4")

    def test_should_delete_item(self):
        # given
        lst = TypedList(*[1, 2, 3], oktypes=(int,))

        # when
        del lst[0]

        # then
        assert len(lst) == 2
        assert 1 not in lst

    def test_should_extend_list(self):
        # given
        lst1 = TypedList(*[1, 2, 3], oktypes=(int,))
        lst2 = TypedList(*[4, 5], oktypes=(int,))

        # when
        lst1.extend(lst2)

        # then
        extended = TypedList(*[1, 2, 3, 4, 5], oktypes=(int,))
        assert lst1 == extended

    @pytest.mark.parametrize("_type, _init, _del, _result", [
        (str, ['a', 'b', 'c'], 'd', ['a', 'b', 'c']),
        (str, ['a', 'b', 'c'], 'b', ['a', 'c']),
        (int, [1, 2, 3], 1, [2, 3]),
    ])
    def test_should_safe_remove_items(self, _type, _init, _del, _result):
        # given
        lst = TypedList(*_init, oktypes=(_type,))

        # when
        lst.safe_remove(_del)

        # then
        result = TypedList(*_result, oktypes=(_type,))
        assert lst == result

    def test_should_sort_list(self):
        # given
        lst = TypedList(*[3, 5, 1, 8], oktypes=(int,))

        # when
        lst.sort(key=lambda el: el)

        # then
        sorted_lst = TypedList(*[1, 3, 5, 8], oktypes=(int,))
        assert lst == sorted_lst
