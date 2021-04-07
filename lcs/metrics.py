def basic_metrics(trial: int, steps: int, reward: int, time: float):
    return {
        'trial': trial,
        'steps_in_trial': steps,
        'reward': reward,
        'perf_time': time,
    }


def population_metrics(population, environment):
    metrics = {
        'population': 0,
        'numerosity': 0,
        'reliable': 0,
    }

    for cl in population:
        metrics['population'] += 1
        metrics['numerosity'] += cl.num
        if cl.is_reliable():
            metrics['reliable'] += 1

    return metrics
