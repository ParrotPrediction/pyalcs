/*
/       (ACS with GA and PEEs inn C++)
/	------------------------------------
/	the Anticipatory Classifier System (ACS) with ALP in action set, GA generalization and PEEs
/
/     (c) by Martin Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 11-30-2000
/
/     effect part class.
*/

#ifndef _Effect_h_
#define _Effect_h_

#include<iostream>
#include<fstream>

#include"Perception.h"
#include"ProbCharPosList.h"
#include"Condition.h"

using namespace std;

class Effect {
public:
    Effect() { list = new ProbCharPosList(); }

    Effect(Effect *eff);

    Effect(Effect *ef1, Effect *ef2, double q1, double q2, Perception *percept);

    ~Effect() { delete list; }

    Perception *getBestAnticipation(Perception *percept);

    Condition *getBestAnticipation(Condition *con);

    int doesAnticipateCorrectly(Perception *p0, Perception *p1);

    int isEnhanced();

    void updateEnhancedEffectProbs(Perception *percept, double updateRate);

    int doesMatch(Perception *percept, Perception *condPercept);

    int doesSpecifyOnlyChangesBackwards(Perception *backAnt, Perception *situation);

    int isEqual(Effect *e2);

    Condition *getAndSpecialize(Perception *p0, Perception *p1);

    int isSpecializable(Perception *p0, Perception *p1);

    int getSpecificity() { return list->getSize(); }

    ProbCharPosList *getList() { return list; }

    friend ostream &operator<<(ostream &out, Effect *e);

private:
    ProbCharPosList *list;
};

#endif
