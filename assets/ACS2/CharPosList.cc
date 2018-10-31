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
/     list structure for condition
*/

#include<iostream>
#include<assert.h>
#include"CharPosList.h"

/**
 * Creates a new list with one item
 */
CharPosList::CharPosList(char chr, int pos) {
    first = new CharPosItem(chr, pos);
    act = first;
    size = 1;
}

/**
 * Creates a copy of the old list 'oldList'.
 */
CharPosList::CharPosList(CharPosList *oldList) {
    CharPosItem *cplp = oldList->first, *cplpNew;

    if (oldList->first != 0) {
        first = new CharPosItem(cplp->c, cplp->p);
        cplpNew = first;
        while (cplp->next != 0) {
            cplp = cplp->next;
            cplpNew->next = new CharPosItem(cplp->c, cplp->p);
            cplpNew = cplpNew->next;
        }
    } else {
        first = 0;
    }
    act = first;
    size = oldList->size;
}

/**
 * Inserts new list object with charcter c and position p in order.
 * Assures that act is not affected except if the first item is inserted.
 * @return If item was successfully inserted.
 */
int CharPosList::insert(char chr, int pos) {
    CharPosItem *cpip, *cpipl;

    //First item
    if (first == 0) {
        first = new CharPosItem(chr, pos);
        size = 1;
        act = first;
        return 1;
    }

    //Look for position in the ordered list
    for (cpip = first, cpipl = 0; cpip != 0; cpip = cpip->next) {
        if (cpip->p >= pos)
            break;
        cpipl = cpip;
    }
    if (cpip != 0 && cpip->p == pos)
        return 0;//Item exists already!

    //Now insert at the determined position
    if (cpipl == 0) {
        first = new CharPosItem(chr, pos);
        first->next = cpip;
    } else {
        cpipl->next = new CharPosItem(chr, pos);
        cpipl->next->next = cpip;
    }
    size++;
    return 1;
}

/**
 * Inserts character 'chr' at the empty position 'epos'. 
 * Assures that act is not affected except if the first item is inserted.
 */
int CharPosList::insertAt(char chr, int epos) {
    CharPosItem *cpip, *cpipl;
    int pos;

    //First item
    if (first == 0) {
        first = new CharPosItem(chr, epos);
        size = 1;
        act = first;
        return 1;
    }
    //Determine pos of empty position
    for (cpip = first, cpipl = 0, pos = -1; cpip != 0; cpip = cpip->next) {
        if (cpip->p - pos - 1 > epos)
            break;
        epos -= (cpip->p - pos - 1);
        pos = cpip->p;
        cpipl = cpip;
    }

    //Now insert!
    if (cpipl == 0) {
        first = new CharPosItem(chr, epos);
        first->next = cpip;
    } else {
        cpipl->next = new CharPosItem(chr, pos + epos + 1);
        cpipl->next->next = cpip;
    }
    size++;
    return 1;
}

/**
 * Inserts list into the current list.
 * Assures that act is not affected except if the first item is inserted.
 */
int CharPosList::insert(CharPosList *list) {
    if (list->size == 0)
        return 0;

    CharPosItem *item = first, *itemL = 0, *listItem;
    for (listItem = list->first; listItem != 0; listItem = listItem->next) {
        //Put item to position after the insert key
        while (item != 0 && item->p < listItem->p) {
            itemL = item;
            item = item->next;
        }
        if (item != 0 && item->p == listItem->p) {
            assert(item->c == listItem->c);/*{
	cerr<<"Tried to specify attribute with different value which is already specified"<<endl;
	return 0;;
	}*/
            size--;
            //Tried to specify item which is already specified but has same value
        } else {
            //Insert item here
            if (itemL == 0) {//Insert in the beginning
                first = new CharPosItem(listItem->c, listItem->p);
                itemL = first;
                itemL->next = item;
            } else {
                itemL->next = new CharPosItem(listItem->c, listItem->p);
                itemL = itemL->next;
                itemL->next = item;
            }
        }
    }
    if (size == 0)
        act = first;
    size += list->size;
    return 1;
}


/**
 * Removes item with key pos.
 */
int CharPosList::remove(int pos) {
    CharPosItem *cpip, *cpipl;
    for (cpip = first, cpipl = 0; cpip != 0; cpip = cpip->next) {
        if (cpip->p == pos)
            break;
        cpipl = cpip;
    }
    if (cpip == 0)//Item not found
        return 0;

    remove(cpip, cpipl);
    return 1;
}

/**
 * Removes nr'st item. (0 init)
 */
int CharPosList::removeAt(int nr) {
    CharPosItem *cpip, *cpipl;
    for (cpip = first, cpipl = 0; nr != 0 && cpip != 0; cpip = cpip->next) {
        nr--;
        cpipl = cpip;
    }

    if (cpip == 0)//Item not found
        return 0;

    remove(cpip, cpipl);
    return 1;
}


/**
 * Removes specified item.
 * Returns if item was found and removed, 0 otherwise.
 */
int CharPosList::removeItem(CharPosItem *item) {
    CharPosItem *cpip, *cpipl;
    for (cpip = first, cpipl = 0; cpip != 0; cpip = cpip->next) {
        if (cpip == item)
            break;
        cpipl = cpip;
    }
    if (cpip == 0)//Item not found
        return 0;

    remove(cpip, cpipl);
    return 1;
}

/**
 * Direct Remover with pointers.
 */
void CharPosList::remove(CharPosItem *cpip, CharPosItem *cpipl) {
    if (cpipl == 0) {
        first = cpip->next;
    } else {
        cpipl->next = cpip->next;
    }
    cpip->next = 0;
    delete cpip;
    size--;
    act = first;
}

/**
 * Returns current CharPosItem and sets act (current) to the next item in the list
 */
CharPosItem *CharPosList::getNextItem() {
    CharPosItem *ret = act;
    if (act != 0)
        act = act->next;
    return ret;
}

/**
 * Realizes uniform crossover between the two charposlists.
 */
void CharPosList::uniformCrossover(CharPosList *cpl2) {
    CharPosItem *cplp = first, *cplpl = 0;
    CharPosItem *cplp2 = cpl2->first, *cplpl2 = 0;

    while (cplp != 0 && cplp2 != 0) {
        if (cplp->p < cplp2->p) {
            if (frand() < 0.5) {
                //switch from this list to the other list
                switchPointer(cplp, cplpl, cpl2, cplp2, cplpl2);
                //set first list to the next place
                if (cplpl == 0)
                    cplp = first;
                else
                    cplp = cplpl->next;
                //set last pointer of second list to the inserted place
                if (cplpl2 == 0)
                    cplpl2 = cpl2->first;
                else
                    cplpl2 = cplpl2->next;
            } else {
                //No change -> go to next candidate in this list
                cplpl = cplp;
                cplp = cplp->next;
            }
        } else if (cplp->p == cplp2->p) {
            if (frand() < 0.5) {
                //switch specific attributes (the values should actually be the same, but we keep it general here)
                char help = cplp->c;
                cplp->c = cplp2->c;
                cplp2->c = help;
            }
            //set both pointers to next candidate
            cplpl = cplp;
            cplp = cplp->next;
            cplpl2 = cplp2;
            cplp2 = cplp2->next;
        } else {
            if (frand() < 0.5) {
                //switch from other list to this list
                cpl2->switchPointer(cplp2, cplpl2, this, cplp, cplpl);
                //set last pointer of first list to the inserted place
                if (cplpl == 0)
                    cplpl = first;
                else
                    cplpl = cplpl->next;
                //set pointer of second list to the next place
                if (cplpl2 == 0)
                    cplp2 = cpl2->first;
                else
                    cplp2 = cplpl2->next;
            } else {
                //No change -> go to next candidate in second list
                cplpl2 = cplp2;
                cplp2 = cplp2->next;
            }
        }
    }

    if (cplp == 0) {
        //Do the remaining stuff in the second list
        while (cplp2 != 0) {
            if (frand() < 0.5) {
                //switch from other list to this list
                cpl2->switchPointer(cplp2, cplpl2, this, cplp, cplpl);
                if (cplpl == 0)
                    cplpl = first;
                else
                    cplpl = cplpl->next;
                if (cplpl2 == 0)
                    cplp2 = cpl2->first;
                else
                    cplp2 = cplpl2->next;
            } else {
                //No change -> move on to the next item in the second list
                cplpl2 = cplp2;
                cplp2 = cplp2->next;
            }
        }
    } else if (cplp2 == 0) {
        //Do the remaining stuff in this list
        while (cplp != 0) {
            if (frand() < 0.5) {
                //switch from this list to the other list
                switchPointer(cplp, cplpl, cpl2, cplp2, cplpl2);
                if (cplpl2 == 0)
                    cplpl2 = cpl2->first;
                else
                    cplpl2 = cplpl2->next;
                if (cplpl == 0)
                    cplp = first;
                else
                    cplp = cplpl->next;
            } else {
                //No change -> move on to the next item in this list
                cplpl = cplp;
                cplp = cplp->next;
            }
        }
    }
}

/**
 * Realizes one-point crossover between the two lists.
 */
void CharPosList::onePointCrossover(CharPosList *cpl2, int len) {
    int point = (int) (frand() * (double) (len - 1));

    CharPosItem *cplp, *cplpl = 0;
    CharPosItem *cplp2, *cplpl2 = 0;
    int s1 = 0, s2 = 0;
    for (cplp = first; cplp != 0; cplp = cplp->next) {
        if (cplp->p > point)
            break;
        s1++;
        cplpl = cplp;
    }
    for (cplp2 = cpl2->first; cplp2 != 0; cplp2 = cplp2->next) {
        if (cplp2->p > point)
            break;
        s2++;
        cplpl2 = cplp2;
    }
    if (cplpl == 0)
        first = cplp2;
    else
        cplpl->next = cplp2;
    if (cplpl2 == 0)
        cpl2->first = cplp;
    else
        cplpl2->next = cplp;
    s1 = size - s1;
    s2 = cpl2->size - s2;
    size = size - s1 + s2;
    cpl2->size = cpl2->size - s2 + s1;
}

/**
 * Realizes two-point crossover between the two lists.
 */
void CharPosList::twoPointCrossover(CharPosList *cpl2, int len) {
    //determine two distinct crossing points and bring them in order
    int p1, p2;
    p1 = (int) (frand() * (double) (len + 1));
    while ((p2 = (int) (frand() * (double) (len + 1))) == p1);//Just make sure that we get two distinct positions
    if (p1 > p2) {
        int help = p1;
        p1 = p2;
        p2 = help;
    }

    //set the pointers of the two lists to the first crossing site
    CharPosItem *cplp, *cplpl = 0;
    CharPosItem *cplp2, *cplpl2 = 0;
    int s1 = 0, s2 = 0;
    for (cplp = first; cplp != 0; cplp = cplp->next) {
        if (cplp->p >= p1)
            break;
        s1++;
        cplpl = cplp;
    }
    for (cplp2 = cpl2->first; cplp2 != 0; cplp2 = cplp2->next) {
        if (cplp2->p >= p1)
            break;
        s2++;
        cplpl2 = cplp2;
    }

    //cross the two lists once
    if (cplpl == 0)
        first = cplp2;
    else
        cplpl->next = cplp2;
    if (cplpl2 == 0)
        cpl2->first = cplp;
    else
        cplpl2->next = cplp;
    if (cplpl == 0)
        cplp = first;
    else
        cplp = cplpl->next;
    if (cplpl2 == 0)
        cplp2 = cpl2->first;
    else
        cplp2 = cplpl2->next;
    //record change in size
    int change = (cpl2->size - s2) - (size - s1);
    size += change;
    cpl2->size -= change;

    //set the pointers to the second crossing site
    for (; cplp != 0; cplp = cplp->next) {
        if (cplp->p >= p2)
            break;
        s1++;
        cplpl = cplp;
    }
    for (; cplp2 != 0; cplp2 = cplp2->next) {
        if (cplp2->p >= p2)
            break;
        s2++;
        cplpl2 = cplp2;
    }
    //cross the two lists again
    if (cplpl == 0)
        first = cplp2;
    else
        cplpl->next = cplp2;
    if (cplpl2 == 0)
        cpl2->first = cplp;
    else
        cplpl2->next = cplp;
    //record the change in size
    change = (cpl2->size - s2) - (size - s1);
    size += change;
    cpl2->size -= change;
}

/**
 * Utility to switch one item from first list to second list.
 */
void CharPosList::switchPointer(CharPosItem *cplp, CharPosItem *cplpl, CharPosList *cpl2, CharPosItem *cplp2,
                                CharPosItem *cplpl2) {
    //delete from one list
    if (cplpl == 0) {
        first = cplp->next;
    } else {
        cplpl->next = cplp->next;
    }
    size--;

    //add to the other list
    cplp->next = cplp2;
    if (cplpl2 == 0) {
        cpl2->first = cplp;
    } else {
        cplpl2->next = cplp;
    }
    cpl2->size++;
}

