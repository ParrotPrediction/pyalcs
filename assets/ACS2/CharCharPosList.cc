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
/     list structure for mark
*/

#include<iostream>

#include"CharList.h"
#include"CharCharPosList.h"

/**
 * Creates a new list with one item
 */
CharCharPosList::CharCharPosList(char chr, int pos)
{
  first=new CharCharPosItem(new CharList(chr), pos);
  act=first;
  size=1;
}

/**
 * Creates a copy of 'oldList'.
 */
CharCharPosList::CharCharPosList(CharCharPosList *oldList)
{
  CharCharPosItem *cplp = oldList->first, *cplpNew;

  if(oldList->first != 0){
    first = new CharCharPosItem(new CharList(cplp->getItem()), cplp->p);
    cplpNew = first;
    while(cplp->next != 0){
      cplp=cplp->next;
      cplpNew->next = new CharCharPosItem(new CharList(cplp->getItem()), cplp->p);
      cplpNew = cplpNew->next;
    }
  }
  act=first;
  size=oldList->size;
}

/**
 * Inserts new list object with charcter c and position p in order.
 * @return If item was successfully inserted.
 */
int CharCharPosList::insert(char chr, int pos)
{
  CharCharPosItem *cpip, *cpipl;

  //First item
  if(first==0){
    first = new CharCharPosItem(new CharList(chr), pos);
    size=1;
    act=first;
    return 1;
  }

  //Look for position in the ordered list
  for(cpip=first, cpipl=0; cpip != 0; cpip=cpip->next){
    if(cpip->p >= pos)
      break;
    cpipl=cpip;
  }
  if(cpip!=0 && cpip->p == pos){
    cpip->item->insert(chr); //Item exists already!->enhance the mark!
  }else{
    //Now insert at the determined position
    if(cpipl == 0){
      first = new CharCharPosItem(new CharList(chr), pos);
      first->next=cpip;
    }else{
      cpipl->next = new CharCharPosItem(new CharList(chr), pos);
      cpipl->next->next = cpip;
    }
    size++;
  }
  act=first;
  return 1;
}

/**
 * Returns current CharCharPosItem and sets act (current) to the next item in the list
 */
CharCharPosItem *CharCharPosList::getNextItem()
{
  CharCharPosItem *ret=act;
  if(act!=0)
    act=act->next;
  return ret;
}

/**
 * Returns item with key pos (if it exists), 0 otherwise
 */
CharCharPosItem *CharCharPosList::getItem(int pos)
{
  for(CharCharPosItem *item=first; item != 0; item=item->next)
    if(item->p==pos)
      return item;
  return 0;
}

