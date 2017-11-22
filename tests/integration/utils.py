def count_macroclassifiers(population):
    return len(population)


def count_microclassifiers(population):
    return sum(cl.num for cl in population)


def count_reliable(population):
    return len([cl for cl in population if cl.is_reliable()])
