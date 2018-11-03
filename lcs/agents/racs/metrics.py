from typing import Dict


def count_averaged_regions(population) -> Dict[int, float]:
    region_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for cl in population:
        for region, counts in cl.get_interval_proportions().items():
            region_counts[region] += counts

    all_elems = sum(i for r, i in region_counts.items())

    return {r: i / all_elems for r, i in region_counts.items()}
