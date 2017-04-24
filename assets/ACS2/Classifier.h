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
/     classifier class.
*/

#ifndef _Classifier_h_
#define _Classifier_h_

#include<fstream>

#include"Condition.h"
#include"Action.h"
#include"Effect.h"
#include"PMark.h"
#include"Perception.h"
#include"ACSConstants.h"
#include"CharPosList.h"
#include"ProbCharPosList.h"

using namespace std;

class Classifier {
public:
    Classifier(Action *action);

    Classifier(Classifier *cl, int time);

    Classifier(Perception *p0, Action *act, Perception *p1, int time);

    ~Classifier() {
        delete C;
        delete A;
        delete E;
        delete M;
    }

    Classifier *expectedCase(Perception *percept, int time);

    Classifier *unexpectedCase(Perception *p0, Perception *p1, int time);

    int isSimilar(Classifier *cl);

    int doesMatch(Perception *percept);

    int doesMatchBackwards(Perception *percept);

    int doesLink(Classifier *cl);

    int hasAction(Action *act) { return act->isEqual(A); }

    int doesAnticipateCorrect(Perception *p0, Perception *p1);

    int doesSubsume(Classifier *cl);

    void mutate();

    void crossover(Classifier *cl);

    Classifier *mergeClassifiers(Classifier *cl2, Perception *percept, int time);

    Perception *getBestAnticipation(Perception *percept);

    Perception *getBackwardsAnticipation(Perception *percept);

    void mark(Perception *p0) { if (M->setMark(C, p0)) ee = 0; }

    double getSpecificity() { return ((double) C->getSpecificity()) / Perception::length; }

    int getUnchangeSpec();

    int doesAnticipateChange();

    void halfQuality() { q /= 2.; }

    void setGATimeStamp(int time) { if (time > tga) tga = time; }

    double setALPTimeStamp(int time);

    double increaseQuality() { return q += BETA * (1. - q); }

    double reverseIncreaseQuality() { return q = (q - BETA) / (1. - BETA); }

    double decreaseQuality() { return q -= BETA * q; }

    double updateReward(double P) { return r += BETA * (P - r); }

    double updateImmediateReward(double rho) { return i += BETA * (rho - i); }

    int increaseNumerosity() { return ++num; }

    int decreaseNumerosity() { return --num; }

    int increaseExperience() { return ++exp; }

    friend ostream &
    operator<<(ostream &out, Classifier *cl);

    Condition *getCondition() { return C; }

    Action *getAction() { return A; }

    Effect *getEffect() { return E; }

    PMark *getPMark() { return M; }

    double getQuality() { return q; }

    double getRewardPrediction() { return r; }

    double getImmediateRewardPrediction() { return i; }

    int getNumerosity() { return num; }

    int getGATimeStamp() { return tga; }

    int getALPTimeStamp() { return talp; }

    double getApplicationAverage() { return tav; }

    int getExperience() { return exp; }

    int isMoreGeneral(Classifier *cl);

    int isSubsumer();

    int isMarked();

    int isEnhanceable() { return ee; }

private:
    Classifier() {
        C = 0;
        A = 0;
        E = 0;
        M = 0;
        q = 0.;
        r = 0.;
        num = 1;
        tga = 0;
        exp = 1;
        ee = 0;
    }

    Classifier(Condition *con, Action *act, Effect *eff);

    Classifier(Condition *con, Action *act, Effect *eff, int time);

    Classifier(Condition *con, Action *act, Effect *eff, int time, double quality, double rewardPrediction,
               double imRewPrediction);

    int generalizeRandomUnchangeCond(int noSpec);

    int
    doesLink(char chr, int pos, CharPosList *cpl2, CharPosItem **cpi2, ProbCharPosList *pcpl2, ProbCharPosItem **pcpi2);

    int doesTightLink(char chr, int pos, CharPosList *cpl2, CharPosItem **cpi2);

    Condition *C;
    Action *A;
    Effect *E;
    PMark *M;
    double q, r, i, tav;
    int num, tga, exp, talp;
    int ee;

    void
    initialize(Condition *con, Action *act, Effect *eff, PMark *mark, int time, double qual, double rew, double iRew,
               double avTime);
};

#endif
