CLASSIFIER_LENGTH = 8
CLASSIFIER_WILDCARD = '#'
NUMBER_OF_EXPERIMENTS = 1
NUMBER_OF_POSSIBLE_ACTIONS = 8

MAX_STEPS = 30000
MAX_TRIAL_STEPS = 50

REWARD_TEST_ITERATION = 50  # Trial steps of printing the performance in a trial

# Algorithm
BETA = 0.05
GAMMA = 0.95
EPSILON = 0.5  # Probability of exploration
THETA_I = 0.1  # Removal in ALP
THETA_EXP = 20  # Classifier experience threshold for determining the subsumer
THETA_R = 0.9  # Classifier quality threshold for determining the subsumer
U_MAX = 100000  # Maximum number of specified elements
