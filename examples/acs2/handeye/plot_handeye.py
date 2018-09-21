# Plot constants
import datetime

import gym
import gym_handeye

from lcs.agents.acs2 import ACS2, ClassifiersList, Configuration
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from examples.acs2.handeye.utils import calculate_performance

TITLE_TEXT_SIZE=24
AXIS_TEXT_SIZE=18
LEGEND_TEXT_SIZE=16


def parse_metrics_to_df(explore_metrics, exploit_metrics):
    def extract_details(row):
        row['trial'] = row['agent']['trial']
        row['steps'] = row['agent']['steps']
        row['numerosity'] = row['agent']['numerosity']
        row['reliable'] = row['agent']['reliable']
        row['knowledge'] = row['performance']['knowledge']
        row['with_block'] = row['performance']['with_block']
        row['no_block'] = row['performance']['no_block']
        return row

    # Load both metrics into data frame
    explore_df = pd.DataFrame(explore_metrics)
    exploit_df = pd.DataFrame(exploit_metrics)

    # Mark them with specific phase
    explore_df['phase'] = 'explore'
    exploit_df['phase'] = 'exploit'

    # Extract details
    explore_df = explore_df.apply(extract_details, axis=1)
    exploit_df = exploit_df.apply(extract_details, axis=1)

    # Adjuts exploit trial counter
    exploit_df['trial'] = exploit_df.apply(
        lambda r: r['trial'] + len(explore_df), axis=1)

    # Concatenate both dataframes
    df = pd.concat([explore_df, exploit_df])
    df.drop(['agent', 'environment', 'performance'], axis=1, inplace=True)
    df.set_index('trial', inplace=True)

    return df


def plot_knowledge(df, ax=None):
    if ax is None:
        ax = plt.gca()

    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")

    explore_df['knowledge'].plot(ax=ax, c='blue')
    explore_df['with_block'].plot(ax=ax, c='green')
    explore_df['no_block'].plot(ax=ax, c='yellow')
    exploit_df['knowledge'].plot(ax=ax, c='red')
    ax.axvline(x=len(explore_df), c='black', linestyle='dashed')

    ax.set_title("Achieved knowledge", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Knowledge [%]", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=LEGEND_TEXT_SIZE)

def plot_classifiers(df, ax=None):
    if ax is None:
        ax = plt.gca()

    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")

    df['numerosity'].plot(ax=ax, c='blue')
    df['reliable'].plot(ax=ax, c='red')

    ax.axvline(x=len(explore_df), c='black', linestyle='dashed')

    ax.set_title("Classifiers", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Classifiers", fontsize=AXIS_TEXT_SIZE)
    ax.legend(fontsize=LEGEND_TEXT_SIZE)


def plot_performance(agent, env, metrics_df, cfg, env_name, additional_info):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'ACS2 Performance in {env_name} environment '
                 f'{additional_info}', fontsize=32)

    # ax1 = plt.subplot(221)
    # plot_policy(env, agent, cfg, ax1)

    ax2 = plt.subplot(211)
    plot_knowledge(metrics_df, ax2)

    ax3 = plt.subplot(212)
    plot_classifiers(metrics_df, ax3)

    # ax4 = plt.subplot(224)
    # plot_steps(metrics_df, ax4)

    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)


def plot_handeye(env_name='HandEye3-v0', filename='images/handeye.pdf', do_action_planning=True):
    hand_eye = gym.make(env_name)
    cfg = Configuration(hand_eye.observation_space.n, hand_eye.action_space.n,
                        epsilon=1.0,
                        do_ga=False,
                        do_action_planning=do_action_planning,
                        performance_fcn=calculate_performance)

    # explore
    agent_he = ACS2(cfg)
    population_maze5_explore, metrics_maze5_explore = agent_he.explore(
        hand_eye, 50)

    # exploit
    agent_he = ACS2(cfg, population_maze5_explore)
    _, metrics_maze5_exploit = agent_he.exploit(hand_eye, 10)

    maze5_metrics_df = parse_metrics_to_df(metrics_maze5_explore,
                                           metrics_maze5_exploit)

    if do_action_planning:
        message = 'with'
    else:
        message = 'without'

    plot_performance(agent_he, hand_eye, maze5_metrics_df, cfg, env_name,
                     '\n{} Action Planning'.format(message))
    plt.savefig(filename, format='pdf', dpi=100)


if __name__ == "__main__":
    env_name = 'HandEye3-v0'

    start = datetime.datetime.now()
    print("time start: {}".format(start))

    plot_handeye(env_name, 'images/{}_ap_{}.pdf'.format(env_name, start),
                 do_action_planning=True)

    middle = datetime.datetime.now()
    print("done with AP, time: {}, elapsed: {}".format(middle, middle-start))

    plot_handeye(env_name, 'images/{}_no_ap_{}.pdf'.format(env_name, start),
                 do_action_planning=False)

    end = datetime.datetime.now()
    print("done without AP, time: {}, elapsed: {}".format(end, end-middle))
