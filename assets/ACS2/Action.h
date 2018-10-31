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
/     action part class header
*/


#ifndef _Action_h_
#define _Action_h_

#include<fstream>
#include<string.h>

using namespace std;

class Environment;

class Action {
public:
    static Environment *env;

    Action() { act = 0; }

    Action(Action *a) { act = a->act; }

    Action(int a) { act = a; }

    ~Action() { ; }

    void setAction(Action *a) { act = a->act; }

    void setAction(int action) { act = action; }

    int isEqual(Action *a2) { if (act == a2->act) return 1; else return 0; }

    int getNr() { return act; }

    friend ostream &operator<<(ostream &out, Action *a);

private:
    int act;
};

#endif
