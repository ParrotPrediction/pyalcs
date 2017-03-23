# File holding all constant values used in learning process

# Don't care symbol
CLASSIFIER_WILDCARD = '#'

# Length of condition and effect part of the classifier (perceptual string).
CLASSIFIER_LENGTH = 4

# Defines a number of possible agents actions
NUMBER_OF_POSSIBLE_ACTIONS = 4

# The 'experience threshold' (natural number) controls when a classifier
# is usable as a subsumer. A low threshold might cause the incorrect
# propagation of an over-general classifier. However, no negative effects
# have been identified so far. Default to 20.
THETA_EXP = 20

# The 'reliability threshold' [0-1] specifies then a classifier is regarded
# as reliable determined by q. The higher the value is set, the longer it
# takes to form a complete model but the more reliable the model actually is.
THETA_R = 0.85

# The 'inadequacy threshold' [0-1]  specifies when a classifier is regarded
# as inadequate (and later removed) determined by its quality q.
THETA_I = 0.1

# The 'specificity threshold' (natural number) - specifies the maximum
# number of specified attributes in C that are anticipated to stay the same
# in E. Used as specialization mechanism ALP. A safe value is always
# CLASSIFIER_LENGTH, the length of the perceptual string.
U_MAX = CLASSIFIER_LENGTH

# FOR ALP
SPEC_ATT = 0.3

# The 'GA application threshold' controls GA frequency (natural number).
# A GA is applied in an action set if the average delay of the last GA
# application of the classifiers in the set is greater than THETA_GA.
# A higher threshold assures that the ALP has enough time to work
# on a generalized set. Default to 100. Lower thresholds usually keep
# the population size down but can cause information loss in the beginning
# of a run.
THETA_GA = 100


# Number of children that will be inserted in the GA process
IN_SIZE = 2

# The 'action set size threshold' (natural number) specifies the maximum
# number of classifiers in an action set which is controlled by the GA.
# Default to 20. If it is set too low, the GA might cause the deletion
# of important classifiers and consequently disrupt the learning process. If
# the size is set very high, the system might learn the problem but it will
# take much longer since the population size will rise a lot.
THETA_AS = 20
