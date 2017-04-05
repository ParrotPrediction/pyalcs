class MazeAction:
    """
    Represents possible actions in Maze environment

    see: https://unicode-table.com/en/sets/arrows-symbols/
    """

    def __init__(self):
        self.actions = {
            'N': {'value': 0, 'symbol': '↑'},
            'NE': {'value': 1, 'symbol': '↗'},
            'E': {'value': 2, 'symbol': '→'},
            'SE': {'value': 3, 'symbol': '↘'},
            'S': {'value': 4, 'symbol': '↓'},
            'SW': {'value': 5, 'symbol': '↙'},
            'W': {'value': 6, 'symbol': '←'},
            'NW': {'value': 7, 'symbol': '↖'},
        }

    def __getitem__(self, item):
        return self.actions[item]

    def get_all_values(self):
        return [v['value'] for k, v in self.actions.items()]

    def find_name(self, value: int):
        for name, mapping in self.actions.items():
            if mapping['value'] == value:
                return name

        raise ValueError('No name for action with value [{}]'.format(value))

    def find_symbol(self, value: int):
        for mapping in self.actions.values():
            if mapping['value'] == value:
                return mapping['symbol']

        raise ValueError('No symbol for action with value [{}]'.format(value))
