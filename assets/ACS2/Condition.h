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
/     condition part class header
*/


#ifndef _Condition_h_
#define _Condition_h_

#include<iostream>
#include<fstream>
#include"CharPosList.h"
#include"Perception.h"

using namespace std;

class Condition {
public:
    Condition() { list = new CharPosList(); }

    Condition(Condition *con) { list = new CharPosList(con->list); }

    ~Condition() { delete list; }

    void specialize(int pos, char c);

    int specialize(Condition *con);

    int generalize(int nr);

    int generalize(CharPosItem *item);

    int doesMatch(Perception *percept);

    int doesMatch(Condition *con);

    int isEqual(Condition *con);

    Perception *getBackwardsAnticipation(Perception *percept);

    int getSpecificity();

    void uniformCrossover(Condition *c2) { list->uniformCrossover(c2->list); }

    void onePointCrossover(Condition *c2, int len) { list->onePointCrossover(c2->list, len); }

    void twoPointCrossover(Condition *c2, int len) { list->twoPointCrossover(c2->list, len); }

    CharPosList *getList() { return list; }

    friend ostream &
    operator<<(ostream &out, Condition *c);

private:
    CharPosList *list;
};

#endif
