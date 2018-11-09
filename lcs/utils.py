def check_types(oktypes, o):
    if not isinstance(o, oktypes):
        raise TypeError(f"Wrong element type: object {o}, type {type(o)}")
