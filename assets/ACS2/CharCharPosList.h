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
/     list structure for mark header
*/

#ifndef _CharCharPosList_h_
#define _CharCharPosList_h_

#include<iostream>

#include"CharList.h"

class CharCharPosItem {
    friend class CharCharPosList;

public:
    int getPos() { return p; }

    CharList *getItem() { return item; }

private:
    CharCharPosItem(CharList *pcl, int pos) {
        p = pos;
        item = pcl;
        next = 0;
    }

    ~CharCharPosItem() {
        delete item;
        delete next;
    }

    int p;
    CharList *item;
    CharCharPosItem *next;
};


class CharCharPosList {
    friend class PMark;

public:
    void reset() { act = first; }

    CharCharPosItem *getNextItem();

    CharCharPosItem *getItem(int pos);

private:
    CharCharPosList() {
        first = 0;
        act = 0;
        size = 0;
    }

    CharCharPosList(char ch, int pos);

    CharCharPosList(CharCharPosList *oldList);

    ~CharCharPosList() { delete first; }

    int insert(char chr, int pos);

    int insertAt(char chr, int epos);

    int getSize() { return size; }

    void remove(CharCharPosItem *cpip, CharCharPosItem *cpipl);

    CharCharPosItem *first;
    CharCharPosItem *act;
    int size;
};

#endif
