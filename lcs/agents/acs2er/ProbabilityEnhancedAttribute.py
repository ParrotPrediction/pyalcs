DETAILED_PEE_ATTR_PRINTING = False


class ProbabilityEnhancedAttribute(dict):
    def __init__(self, attr):
        assert isinstance(attr, str) or isinstance(attr, dict)
        super().__init__()

        if isinstance(attr, str):
            self[attr] = 1.0

        if isinstance(attr, dict):
            for symbol in attr:
                self[symbol] = attr[symbol]

        self.adjust_probabilities()

    @classmethod
    def merged_attributes(cls,
                          attr1,
                          attr2,
                          q1: float = 0.5,
                          q2: float = 0.5):
        """
        Create a new enhanced effect part.
        """

        result = attr1.copy() \
            if isinstance(attr1, ProbabilityEnhancedAttribute) \
            else ProbabilityEnhancedAttribute(attr1)

        result.insert(attr2, q1, q2)
        return result

    def sum_of_probabilities(self):
        return sum(self.get(symbol, 0.0) for symbol in self)

    def adjust_probabilities(self, prev_sum=None):
        """
        Adjust the probabilities to sum to one
        """
        if prev_sum is None:
            prev_sum = self.sum_of_probabilities()

        for symbol in self:
            self[symbol] /= prev_sum

    def increase_probability(self, effect_symbol, update_rate):
        if effect_symbol not in self:
            return False

        update_delta = update_rate * (1 - self[effect_symbol])
        self[effect_symbol] += update_delta
        self.adjust_probabilities(1.0 + update_delta)
        return True

    def get_best_symbol(self):
        """
        Equivalent to ProbCharList::getBestChar() in C++ code.
        """
        return max(self.items(), key=lambda x: x[1])[0]

    def does_contain(self, symbol):
        """
        Checks whether the specified symbol occurs in the attribute.
        """
        return self.get(symbol, 0.0) != 0.0

    def is_enhanced(self):
        """
        The attribute is enhanced if it's not reduced to a single symbol
        with probability 100%.
        """
        return sum(1 for k, v in self.items() if v > 0.0) > 1

    def the_only_symbol(self):
        assert not self.is_enhanced()
        # In case of non enhanced attribute, the only symbol is the best symbol
        return self.get_best_symbol()

    def symbols_specified(self):
        return {k for k, v in self.items() if v > 0.0}

    def is_similar(self, other):
        """
        Determines if the two lists specify the same characters.
        Order and probabilities are not considered.
        """
        if isinstance(other, ProbabilityEnhancedAttribute):
            return self.symbols_specified() == other.symbols_specified()
        else:
            return self.symbols_specified() == {other}

    def is_compact(self):
        return not any(prob == 0.0 for symbol, prob in self.items())

    def make_compact(self):
        for symbol, prob in list(self.items()):
            if prob == 0.0:
                del self[symbol]

    def insert_symbol(self, symbol, q1=1.0, q2=None):
        if q2 is None:
            q2 = 1.0 / len(self)

        for sym in self:
            self[sym] *= q1
        self[symbol] = self.get(symbol, 0.0) + q2
        self.adjust_probabilities()

    def insert_attribute(self, o, q1, q2):
        assert isinstance(o, ProbabilityEnhancedAttribute)

        for symbol in self.symbols_specified().union(o.symbols_specified()):
            self[symbol] = self.get(symbol, 0.0) * q1 + o.get(symbol, 0.0) * q2
            self.adjust_probabilities()

    def insert(self, symbol_or_attr, q1, q2):
        if isinstance(symbol_or_attr, ProbabilityEnhancedAttribute):
            self.insert_attribute(symbol_or_attr, q1, q2)
        else:
            self.insert_symbol(symbol_or_attr, q1, q2)

    def remove_symbol(self, symbol):
        if symbol in self:
            symbols = self.symbols_specified()
            symbols.remove(symbol)
            if len(symbols) == 0:
                # Refuse to remove the last symbol
                return False
            del self[symbol]
            self.adjust_probabilities()
            return True
        else:
            return False

    def copy(self):
        return ProbabilityEnhancedAttribute(self)

    def sorted_items(self):
        return sorted(self.items(), key=lambda x: x[1], reverse=True)

    def __eq__(self, other):
        return self.is_similar(other)

    def __str__(self):
        if DETAILED_PEE_ATTR_PRINTING:
            return "{" + ", ".join(
                "%s:%.0f%%" % (sym, sym[1] * 100)
                for sym in self.sorted_items()) + "}"
        else:
            return "{" + "".join(str(sym[0])
                                 for sym in self.sorted_items()) + "}"
