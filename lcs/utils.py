def check_types(oktypes, o):
    if not isinstance(o, oktypes):
        raise TypeError(
            "Wrong element type: object {}, type {}".format(o, type(o)))
