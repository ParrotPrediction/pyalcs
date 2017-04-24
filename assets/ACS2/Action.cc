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
/     action part class.
*/

#include<iostream>
#include"Action.h"
#include"Environment.h"

using namespace std;

ostream &operator<<(ostream &out, Action *a) {
    char *action = a->env->getActionString(a);
    out << action;
    delete[] action;
    return out;
}
