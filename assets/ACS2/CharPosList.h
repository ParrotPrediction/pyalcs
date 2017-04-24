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
/     list structure for condition header
*/


#ifndef _CharPosList_h_
#define _CharPosList_h_

#include<iostream>
#include<stdlib.h>

#include"Perception.h"


class CharPosItem {
    friend class CharPosList;

public:
    int getPos() { return p; }

    char getChar() { return c; }

private:
    CharPosItem(char chr, int pos) {
        p = pos;
        c = chr;
        next = 0;
    }

    ~CharPosItem() { delete next; }

    int p;
    char c;
    CharPosItem *next;
};


class CharPosList {
public:
    CharPosList() {
        first = 0;
        act = 0;
        size = 0;
    }

    CharPosList(char ch, int pos);

    CharPosList(CharPosList *oldList);

    ~CharPosList() { delete first; }

    char getChar(int pos);

    int insert(char chr, int pos);

    int insertAt(char chr, int epos);

    int insert(CharPosList *list);

    int remove(int pos);

    int removeAt(int nr);

    int removeItem(CharPosItem *item);

    int getSize() { return size; }

    void reset() { act = first; }

    void uniformCrossover(CharPosList *cpl);

    void onePointCrossover(CharPosList *cpl, int len);

    void twoPointCrossover(CharPosList *cpl, int len);

    CharPosItem *getNextItem();

private:
    void remove(CharPosItem *cpip, CharPosItem *cpipl);

    void
    switchPointer(CharPosItem *cplp, CharPosItem *cplpl, CharPosList *cpl2, CharPosItem *cplp2, CharPosItem *cplpl2);

    CharPosItem *first;
    CharPosItem *act;
    int size;
};

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0))
#endif

#endif
