/*
/       ACS2 in C++
/	------------------------------------
/       choice of without/with GA, subsumption, PEEs
/
/     (c) by Martin V. Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 02-23-2001
/
/     blocks world environment class header
*/

#ifndef _bw_environment_h_
#define _bw_environment_h_

#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#include"Environment.h"
#include"Perception.h"
#include"Action.h"

#define NR_BLOCKS 4 /* = perceptual hight of one stack */
#define NR_STACKS 4 /* the number of different available/distinguishable stacks on the table */
#define NR_DIFFERENT_BLOCKS 1 /* if >1 colors are chosen probabilistically assuring the number of different blocks */

#define BW_TEST_NO 100 /* number of tests during testing mode */
#define BW_TEST_ONLY_CHANGES 0 /* defines if only changes or all possibilities should be tested (-1 = only non-changes)*/
#define BW_TEST_ALL_POSSIBILITIES_UNIFORMLY 0 /*defines if all possibilities should be tested uniformly randomly during testing */

class BWEnvironment : public Environment {
public:
    BWEnvironment(char *nothing);

    ~BWEnvironment();

    void getSituation(Perception *perception);

    int getPerceptionLength() { return envSize; }

    double executeAction(Action *act);

    int isReset();

    int reset();

    int getNoActions();

    Action **getActions();

    char *getActionString(Action *act);

    void doTesting();//Used for testing purposes
    int getNextTest(Perception *p0, Action *act, Perception *p1);

    void endTesting();//Used for testing purposes
    int getGoalState(Perception *perception) { return 0; }

    char *getID() {
        char *id = new char[3];
        strcpy(id, "BW");
        return id;
    }

    friend ostream &
    operator<<(ostream &out, BWEnvironment *env);

private:
    void resetUniform();

    char *env;
    char *originalEnv;
    int envSize;
    int testCounter;
};

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0))
#endif

#endif

