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
/     list structure of one attribute in the mark
*/

#include<iostream>

#include"CharList.h"

/**
 * Creates a copy of oldList.
 */
CharList::CharList(CharList *oldList) {
    if (oldList != 0) {
        c = oldList->c;
        if (oldList->next != 0)
            next = new CharList(oldList->next);
        else
            next = 0;
    }
}

/**
 * Inserts character c if not yet existent.
 * If c does exist alread return 0. 
 * Returns if insertion was successful, 0 otherwise.
 */
int CharList::insert(char c) {
    CharList *clp = this;

    for (clp = this; clp != 0; clp = clp->next) {
        if (clp->c == c) {
            return 0;
        }
    }
    //Charcter does not exist yet->add after first character in the list
    clp = next;
    next = new CharList(c);
    next->next = clp;
    return 1;
}

/**
 * Removes character 'chr' from this charlist.
 * Returns if character was found and removed, 0 otherwise.
 */
int CharList::remove(char chr) {
    CharList *clp, *clpl = 0;
    for (clp = this; clp != 0; clp = clp->next) {
        if (clp->c == chr)
            break;
        clpl = clp;
    }

    if (clp == 0)// character does not exist!
        return 0;

    if (clpl == 0) {//Delete first item -> copy values of second item and delete second item!
        if (next == 0)//Never Delete the last item!
            return 0;
        c = next->c;
        clp = next;
        next = next->next;
        clp->next = 0;
        delete clp;
    } else {
        clpl->next = clp->next;
        clp->next = 0;
        delete clp;
    }
    return 1;
}

/**
 * Returns if the list contains character 'testC'.
 */
int CharList::doesContain(char testC) {
    for (CharList *listp = this; listp != 0; listp = listp->next) {
        if (listp->c == testC)
            return 1;
    }
    return 0;
}

/**
 * Returns if both lists contain the same characters.
 */
int CharList::isIdentical(CharList *list2) {
    if (!isEnhanced())
        if (list2->isEnhanced()) {
            return 0;
        } else {
            if (c != list2->c)
                return 0;
        }
    else if (!list2->isEnhanced()) {
        return 0;
    } else {
        //both lists are enhanced -> check from both sides
        CharList *listp1, *listp2;
        for (listp1 = this; listp1 != 0; listp1 = listp1->next) {
            for (listp2 = list2; listp2 != 0; listp2 = listp2->next)
                if (listp2->c == listp1->c)
                    break;
            if (listp2 == 0)
                return 0;
        }
        for (listp2 = list2; listp2 != 0; listp2 = listp2->next) {
            for (listp1 = this; listp1 != 0; listp1 = listp1->next)
                if (listp2->c == listp1->c)
                    break;
            if (listp1 == 0)
                return 0;
        }
    }
    return 1;
}

ostream &operator<<(ostream &out, CharList *cl) {
    if (cl->isEnhanced()) {
        out << "{";
        for (CharList *listp = cl; listp != 0; listp = listp->next) {
            out << listp->c;
            if (listp->next != 0)
                out << ',';
        }
        out << "}";
    } else {
        out << cl->c;
    }
    return out;
}


