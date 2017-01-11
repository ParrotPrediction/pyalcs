class MazeMapping:
    """
    Represents possible symbols and corresponding perception values for maze definitions
    """

    def __init__(self):
        self.mapping = {}
        self.mapping['wall'] = {'symbol': '#', 'value': -1}
        self.mapping['path'] = {'symbol': '.', 'value': 0}
        self.mapping['reward'] = {'symbol': '$', 'value': 1}

    def find_value(self, symbol: str) -> int:
        """
        Returns value for the given maze symbol. The value is used for animat perception

        :param symbol: symbol from maze definition
        :return: an integer mapping symbol
        """
        for v in self.mapping.values():
            if v['symbol'] == symbol:
                return v['value']

        raise ValueError('No mapping for symbol ' + symbol)