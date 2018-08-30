from bitstring import BitString


def get_correct_answer(bitstring: list, control_bits: int) -> int:
    bits = BitString(bitstring)

    _ctrl_bits = bits[:control_bits]
    _data_bits = bits[control_bits:]

    return int(_data_bits[_ctrl_bits.uint])
