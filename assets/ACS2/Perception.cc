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
/     perception class
*/

#include<iostream>
#include<assert.h>
#include <cstring>
#include"Perception.h"

/**
 * Creates a copy of the old perception 'old'.
 */
Perception::Perception(Perception *old) {
    percept = new char[strlen(old->percept) + 1];
    strcpy(percept, old->percept);
}

/**
 * Sets the perception to the string 'in'.
 */
void Perception::setPerception(char *in) {
    assert((signed int) strlen(in) == length);
    strcpy(percept, in);
}

/**
 * Copies the perception 'percept' to this perception.
 */
void Perception::setPerception(Perception *percept) {
    strcpy(this->percept, percept->percept);
}

/**
 * Returns if the two perceptions are equal.
 */
int Perception::isEqual(Perception *perception) {
    if (strcmp(percept, perception->percept) == 0)
        return 1;
    else
        return 0;
}

ostream &operator<<(ostream &out, Perception *p) {
    out << p->percept;
    return out;
}
