/*
/       (ACS with GA and PEEs in C++)
/	------------------------------------
/	the Anticipatory Classifier System (ACS) with ALP in action set, GA generalization and PEEs
/
/     (c) by Martin Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 11-30-2000
/
/     Multiplexer environment class header
*/


#ifndef _mp_environment_h_
#define _mp_environment_h_

#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#include"Environment.h"
#include"Perception.h"
#include"Action.h"

#define MULTIPLEXER 10
#define PAYOFFMAP_IN_PERCEPTIONS 1
#define WHICHCODE0112 0
#define RETURN_REWARDMAP 0
#define MP_TEST_SIZE 100

class MPEnvironment : public Environment {
public:
    MPEnvironment(char *fileName);

    ~MPEnvironment();

    void getSituation(Perception *perception);

    int getPerceptionLength() { return condLength; }

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
        strcpy(id, "MP");
        return id;
    }

    friend ostream &
    operator<<(ostream &out, MPEnvironment *env);

private:
    int doReset;
    int testNr;
    int testing;
    int posbits;
    int rcodeLength;
    int condLength;
    char *cond;
    char *testCond;
};

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0))
#endif

#endif
