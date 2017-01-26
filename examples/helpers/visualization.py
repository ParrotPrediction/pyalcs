import matplotlib.pyplot as plt


def plot_performance(**kwargs):

    # Extract arguments
    time = kwargs.pop('time')
    total_classifiers = kwargs.pop('total_classifiers')
    f_reward = kwargs.pop('found_reward')
    avg_quality = kwargs.pop('average_quality')
    avg_fitness = kwargs.pop('average_fitness')

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.plot(time, total_classifiers, 'b')
    ax1.plot(time, _filter(total_classifiers, f_reward), 'r.')
    ax1.set_title('Total Classifiers')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Macro-classifiers')
    ax1.grid(True)

    ax2 = fig.add_subplot(222)
    ax2.plot(time, avg_quality, 'g')
    ax2.plot(time, _filter(avg_quality, f_reward), 'r.')
    ax2.set_title('Quality')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Quality per classifier')
    ax2.grid(True)

    ax3 = fig.add_subplot(223)
    ax3.plot(time, avg_fitness, 'y')
    ax3.plot(time, _filter(avg_fitness, f_reward), 'r.')
    ax3.set_title('Fitness')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Fitness per classifier')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('learning_progress.png', dpi=200)


def _filter(metrics: list, mask: list) -> list:
    filtered = []

    for i, n in enumerate(metrics):
        if mask[i]:
            filtered.append(n)
        else:
            filtered.append(None)

    return filtered
