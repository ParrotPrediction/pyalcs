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
/     list structure for one attribute in the effect part (essential for PEEs)
*/

#include<iostream>

#include"ProbCharList.h"

using namespace std;

/**
 * Creates a copy of the old ProbCharList 'oldList'.
 */
ProbCharList::ProbCharList(ProbCharList *oldList) {
    if (oldList != 0) {
        p = oldList->p;
        c = oldList->c;
        if (oldList->next != 0)
            next = new ProbCharList(oldList->next);
        else
            next = 0;
    }
}

/**
 * Increases the probablity of the character ch by the rate updateRate and relveals the overall probablity.
 * Returns if the character was actually present.
 */
int ProbCharList::increaseProbability(char ch, double updateRate) {
    ProbCharList *listP = this;
    while (listP != 0 && listP->c != ch)
        listP = listP->next;
    if (listP == 0)
        return 0; //Character was not found in this ProbCharList
    double update = updateRate * (1 - listP->p);
    listP->p += update;
    adjustProbabilities(1. + update);
    return 1;
}

/**
 * Inserts a whole other ProbCharList.
 * Assures that characters are not represented twice!
 * q1 and q2 represent the qualities of the corresponding classifiers and are weighted in
 */
void ProbCharList::insert(ProbCharList *merger, double q1, double q2) {
    ProbCharList *mergeList = new ProbCharList(merger);

    ProbCharList *pclp, *pclpnew;
    for (pclp = this; pclp != 0; pclp = pclp->next) {
        pclp->p *= q1;
        for (pclpnew = mergeList; pclpnew != 0; pclpnew = pclpnew->next) {
            if (pclp->c == pclpnew->c) {
                pclp->p += (pclpnew->p * q2);
                pclpnew->p = 0; //-> serves as a reminder that this one was inserted already
                break;
            }
        }
    }

    for (pclp = this; pclp->next != 0; pclp = pclp->next); //Set pclp to the end of the list

    for (pclpnew = mergeList; pclpnew != 0; pclpnew = pclpnew->next) {
        if (pclpnew->p != 0) {
            pclp->next = new ProbCharList(pclpnew->c);
            pclp = pclp->next;
            pclp->p = pclpnew->p * q2;
        }
    }
    delete mergeList;
    adjustProbabilities();
}

/**
 * Inserts Character c if not yet existent. Sets probability of c to the average.
 */
void ProbCharList::insert(char c) {
    double n = 0;
    ProbCharList *pclp;
    for (pclp = this; pclp != 0; pclp = pclp->next)
        n++;

    insert(c, 1., 1. / n);
}

/**
 * Inserts Character c if not yet existent.
 * If c does exist alread, add prob to the probability and adjust probabilities.
 * If it does not exist yet, add it to the list, set the prob to prob and adjust probabilities. 
 * q1 and q2 represent the qualities of the corresponding classifiers and are weighted in
 */
void ProbCharList::insert(char c, double q1, double q2) {
    // although pSum should be one, I decided to recalculate it here!
    double pSum = 0;
    ProbCharList *pclp = this;
    int found = 0;

    // Look for Character
    for (pclp = this; pclp != 0; pclp = pclp->next) {
        pclp->p *= q1;
        pSum += pclp->p;
        if (pclp->c == c) {
            pclp->p += q2;
            pSum += q2;
            adjustProbabilities();
            found = 1;
        }
    }
    if (!found) {
        //Charcter does not exist yet -> insert in after this guy
        pclp = next;
        next = new ProbCharList(c);
        next->next = pclp;
        next->p = q2;
        pSum += q2;
    }
    adjustProbabilities(pSum);
}

/**
 * Removes character 'chr' from the list if present. Adjusts probabilities of remaining items accordingly.
 * Note that the last character is never deleted (which should never occur anyways).
 * Returns one if the character was found and removed, 0 otherwise.
 */
int ProbCharList::remove(char chr) {
    ProbCharList *pclp, *pclpl = 0;
    //Look for character
    for (pclp = this; pclp != 0; pclp = pclp->next) {
        if (pclp->c == chr)
            break;
        pclpl = pclp;
    }

    if (pclp == 0)// character does not exist!
        return 0;

    if (pclpl == 0) {//Delete first item -> copy values of second item and delete second item!
        if (next == 0)//Never Delete the last item!
            return 0;
        p = next->p;
        c = next->c;
        pclp = next;
        next = next->next;
        pclp->next = 0;
        delete pclp;
    } else {
        pclpl->next = pclp->next;
        pclp->next = 0;
        delete pclp;
    }
    adjustProbabilities();
    return 1;
}

/**
 * Determines the character with the highest probability in the list.
 * Returns the character witht the highest associated probability.
 */
char ProbCharList::getBestChar() {
    double bestP = 0.0;
    char bestC = '\0';
    for (ProbCharList *listp = this; listp != 0; listp = listp->next) {
        if (listp->p > bestP) {
            bestP = listp->p;
            bestC = listp->c;
        }
    }
    return bestC;
}

/**
 * Determines if the list contains character 'testC'
 * Returns one if the character is contained, 0 otherwise.
 */
int ProbCharList::doesContain(char testC) {
    for (ProbCharList *listp = this; listp != 0; listp = listp->next) {
        if (listp->c == testC)
            return 1;
    }
    return 0;
}

/**
 * Determines if the two lists specify the same characters. 
 * Order and probabilities are not considered.
 */
int ProbCharList::isSimilar(ProbCharList *list2) {
    if (!isEnhanced()) {
        if (list2->isEnhanced()) {
            return 0;
        } else {
            if (c != list2->c)
                return 0;
        }
    } else {
        if (!list2->isEnhanced()) {
            return 0;
        } else {
            ProbCharList *listp1, *listp2;
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
    }
    return 1;
}

/**
 * Adjusts the probabilities in the list to sum to one.
 */
void ProbCharList::adjustProbabilities() {
    ProbCharList *pclp;
    double pSum = 0.;
    for (pclp = this; pclp != 0; pclp = pclp->next)
        pSum += pclp->p;
    adjustProbabilities(pSum);
}

/**
 * Adjusts the probabilities in the list to sum to one given the current sum of the probabilities.
 */
void ProbCharList::adjustProbabilities(double pSum) {
    ProbCharList *pclp;
    for (pclp = this; pclp != 0; pclp = pclp->next)
        pclp->p /= pSum;
}


ostream &operator<<(ostream &out, ProbCharList *pcl) {
    if (pcl->isEnhanced()) {
        out << " {";
        for (ProbCharList *listp = pcl; listp != 0; listp = listp->next) {
            out << "(" << listp->c << ")";//,"<<listp->p<<")";
            if (listp->next != 0)
                out << ',';
        }
        out << "} ";
    } else {
        out << pcl->c;
    }
    return out;
}


