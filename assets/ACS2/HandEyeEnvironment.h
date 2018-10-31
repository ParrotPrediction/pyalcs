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
/     hand-eye gripper environment class header
*/

#ifndef _handeye_environment_h_
#define _handeye_environment_h_

#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#include"Environment.h"
#include"Perception.h"
#include"Action.h"

#define HE_GRID_SIZE 3 /* x size of the 2-D quadratic plane */
#define NOTE_IN_HAND 1 /*specifies if feeler should switch to '2' when block is in hand */

#define HE_TEST_NO 100 /* number of tests during testing mode */
#define HE_TEST_ONLY_CHANGES 0 /* defines if only changes, non changes, or all possibilities should be tested */

class HandEyeEnvironment : public Environment {
public:
    HandEyeEnvironment(char *nothing);

    ~HandEyeEnvironment();

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
    int getGoalState(Perception *perception);

    char *getID() {
        char *id = new char[8];
        strcpy(id, "HandEye");
        return id;
    }

    friend ostream &
    operator<<(ostream &out, HandEyeEnvironment *env);

private:
    double moveGripper(int xstart, int ystart, int xend, int yend);

    double gripBlock(int x, int y);

    double releaseBlock(int x, int y);

    int envSize;

    char *env;
    char *originalEnv;

    int gPosX;
    int gPosY;
    int originalGPosX;
    int originalGPosY;

    int blockInHand;
    int originalBlockInHand;

    int *blockPositions;
    int *originalBlockPositions;

    int testCounter;

    int goalGeneratorState;
};

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0))
#endif

#endif

