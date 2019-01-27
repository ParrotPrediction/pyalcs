def check_types(oktypes, o):
    if type(o) not in oktypes:
        raise TypeError(f"Wrong element type: object {o}, type {type(o)}")
