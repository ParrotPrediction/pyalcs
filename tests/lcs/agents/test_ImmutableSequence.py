from lcs.agents import ImmutableSequence


class TestImmutableSequence:
    def test_should_hash(self):
        assert hash(ImmutableSequence('111')) == hash(ImmutableSequence('111'))
        assert hash(ImmutableSequence('111')) != hash(ImmutableSequence('112'))
