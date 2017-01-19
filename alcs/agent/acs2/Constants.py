# File holding all constant values used in learning process

# Don't care symbol
CLASSIFIER_WILDCARD = '#'

# Length of condition and effect part of the classifier
CLASSIFIER_LENGTH = 4

# Defines a number of possible agents actions
NUMBER_OF_POSSIBLE_ACTIONS = 4

# The exploration probability [0-1]. Specifies the probability of choosing
# a random action.
EPSILON = 0.5

# Threshold of required classifier experience to subsume another classifier
THETA_EXP = 20

# The 'reliability threshold' [0-1] specifies then a classifier is regarded
# as reliable determined by q. The higher the value is set, the longer it
# takes to form a complete model but the more reliable the model actually is.
THETA_R = 0.9

# The 'inadequacy threshold' [0-1]  specifies when a classifier is regarded
# as inadequate (and later removed) determined by its quality q.
THETA_I = 0.1

# The 'learning rate' - used in ALP and RL. Updates affecting q, r, ir, aav.
# The usual value (passive is 0.05). The higher, the faster parameters approach
# an approximation of their actual value but the more noisy the approximation
# is.
BETA = 0.2

# For ALP
U_MAX = 6

# FOR ALP
SPEC_ATT = 0.3

# For RL
GAMMA = 0.95

# 'Application threshold' controls GA frequency. A GA is applied in an action
# set if the average delay of the last GA application of the classifiers in
# the set is greater than THETA_GA. Can be any natural number.
THETA_GA = 5

# The 'crossover probability' [0-1] specifies the probability of applying
# crossover in the conditions of the offspring when a GA is applied.
X = 0.8

# For GA
MU = 0.1

# For GA
IN_SIZE = 3

THETA_AS = 0.5
