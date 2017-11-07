/*
/       ACS2 in C++
/	------------------------------------
/       choice without/with GA, subsumption, PEEs
/
/     (c) by Martin V. Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 03-04-2001
/
/     main program
*/

#ifndef _ACSConstants_h_
#define _ACSConstants_h_

/*Parameter:*/
#define BETA 0.05
#define GAMMA 0.95

#define THETA_I 0.1
#define THETA_R 0.9

#define R_INI 0.5
#define IR_INI 0
#define Q_INI 0.5
#define AVT_INI 0

#define Q_ALP_MIN 0.5
#define Q_GA_DECREASE 0.5

#define U_MAX 100000

#define DO_PEES 0

#define DO_MENTAL_ACTING_STEPS 0
#define DO_LOOKAHEAD_WINNER 0

#define EPSILON 1.0  /* Probability of Exploration */

#define PROB_EXPLORATION_BIAS 0.5 /* specifies the probability of applying an exploration biased action-selection */
#define EXPLORATION_BIAS_METHOD 2 /* 0 = action delay bias, 1 = knowledge array bias, 2 = 50/50 */

#define DO_ACTION_PLANNING 0
#define ACTION_PLANNING_FREQUENCY 50

/* GA constants */
#define DO_GA 0
#define THETA_GA 100
#define MU 0.30
#define X_TYPE 2 /* 0 = uniform, 1 = one-point, and 2 = two-point crossover */
#define CHI 0.8
#define THETA_AS 20
#define THETA_EXP 20

#define DO_SUBSUMPTION 1

/*andere Makros:*/
#define ENVIRONMENT_CLASS MazeEnvironment
#define RESULT_FILE "ACS2_Maze4_5050B050.txt"

#define MAX_STEPS 30000
#define MAX_TRIAL_STEPS 50
#define ANZ_EXPERIMENTS 20

#define REWARD_TEST 0
#define MODEL_TEST_TYPE 0 /* 0 = test all reliable classifiers, 1 = test highest quality classifier */

#define MODEL_TEST_ITERATION 200 /* Test Iterations in the latent learning tests */
#define REWARD_TEST_ITERATION 50 /* Trial steps of printing the performance in a trial */

#define DONT_CARE '#'

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0))
#endif

#endif





