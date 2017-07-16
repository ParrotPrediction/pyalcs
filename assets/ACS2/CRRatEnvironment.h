/*
/       ACS2 in C++
/	------------------------------------
/       choice of without/with GA, subsumption, PEEs
/
/     (c) by Martin V. Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 03-17-2001
/
/     collwill and rescorla's rat experiments
*/


#ifndef _crrat_environment_h_
#define _crrat_environment_h_

#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#include"Environment.h"
#include"Perception.h"
#include"Action.h"

#define CR_DO90 1

#define CR85_PHASE1TIME 204
#define CR85_PHASE2TIME 100
#define CR85_PHASE3TIME 50

#define CR90_PHASE1TIME 64
#define CR90_PHASE2TIME 176
#define CR90_PHASE3TIME 100
#define CR90_PHASE4TIME 80

class CRRatEnvironment : public Environment {
public:
    CRRatEnvironment(char *fileName);

    ~CRRatEnvironment();

    void getSituation(Perception *perception);

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

    int getPerceptionLength() { return length; }

    char *getID() {
        char *id = new char[6];
        strcpy(id, "CRRat");
        return id;
    }

    friend ostream &
    operator<<(ostream &out, CRRatEnvironment *env);

private:
    int timer;
    int phase;
    int length;
    int doReset;
    char *situation;
};

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0))
#endif

#endif
