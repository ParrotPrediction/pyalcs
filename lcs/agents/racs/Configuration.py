from typing import Callable, Optional

from lcs.representations import UBR


class Configuration:
    def __init__(self, **kwargs) -> None:
        self.classifier_length: int = kwargs['classifier_length']
        self.number_of_possible_actions: int = kwargs[
            'number_of_possible_actions']
        self.encoder = kwargs['encoder']

        self.beta: float = kwargs.get('beta', 0.05)
        self.gamma: float = kwargs.get('gamma', 0.95)
        self.theta_i: float = kwargs.get('theta_i', 0.1)
        self.theta_r: float = kwargs.get('theta_r', 0.9)
        self.epsilon: float = kwargs.get('epsilon', 0.5)
        self.biased_exploration: float = kwargs.get('biased_exploration', 0.5)

        # Max range of uniform noise distribution that can alter
        # the perception during covering U[0, cover_noise]
        self.cover_noise: float = kwargs.get('cover_noise', 0.1)
        self.u_max: int = kwargs.get('u_max', 100000)
        self.do_subsumption: bool = kwargs.get('do_subsumption', True)
        self.do_ga: bool = kwargs.get('do_ga', False)
        self.mu: float = kwargs.get('mu', 0.3)

        # Max range of uniform noise distribution that can broaden the
        # phenotype interval range.
        self.mutation_noise: float = kwargs.get('mutation_noise', 0.1)
        self.chi: float = kwargs.get('chi', 0.8)
        self.theta_exp: int = kwargs.get('theta_exp', 20)
        self.theta_ga: int = kwargs.get('theta_ga', 100)
        self.theta_as: int = kwargs.get('theta_as', 20)

        self.metrics_trial_frequency: int = kwargs.get(
            'metrics_trial_frequency', 5)
        self.user_metrics_collector_fcn: Optional[Callable] = kwargs.get(
            'user_metrics_collector_fcn', None)
        self.model_checkpoint_freq: Optional[int] = kwargs.get(
            'model_checkpoint_frequency', None)

        self.oktypes = (UBR,)
        self.classifier_wildcard = UBR(*self.encoder.range)
