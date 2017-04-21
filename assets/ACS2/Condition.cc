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
/     condition part class.
*/

#include<iostream>
#include<fstream>

#include"Perception.h"
#include"CharPosList.h"
#include"Condition.h"

/**
 *Specializes the condition on position pos with character c
 */
void Condition::specialize(int pos, char c)
{
  list->insert(c, pos);
}

/**
 * Specializes all in 'con' specialized attributes.
 */
int Condition::specialize(Condition *con)
{
  return list->insert(con->list);
}

/**
 * Generalizes the 'nr''th specific attribute.
 */
int Condition::generalize(int nr)
{
  return list->removeAt(nr);
}

/**
 * Generalizes the specified attribute.
 */
int Condition::generalize(CharPosItem *item)
{
  return list->removeItem(item);
}

/**
 * Returns if condition matches perception 
 */
int Condition::doesMatch(Perception *percept)
{
  list->reset();
  CharPosItem *item=list->getNextItem();
  
  while(item != 0) {
    if( percept->getAttribute(item->getPos()) != item->getChar() )
      return 0;
    item = list -> getNextItem();
  }
  return 1;
}

/**
 * Returns if condition matches other condition.
 */
int Condition::doesMatch(Condition *con)
{
  list->reset();
  con->list->reset();
  CharPosItem *item1=list->getNextItem();
  CharPosItem *item2=con->list->getNextItem();
  while(item1 != 0 && item2 != 0) {
    if( item1->getPos() < item2->getPos()){ 
      item1 = list->getNextItem();
    }else if(item1->getPos() == item2->getPos()){
      if(item1->getChar() != item2->getChar()){
	return 0;
      }else{
	item1 = list->getNextItem();
	item2 = con->list->getNextItem();
      }
    }else{
      item2 = con->list->getNextItem();
    }
  }
  //Rest in one matches don't cares in others. Thus, we can return true!
  return 1;
}

/**
 * Returns if condition is identical to other condition.
 */
int Condition::isEqual(Condition *con)
{
  list->reset();
  con->list->reset();
  
  if(list->getSize() != con->list->getSize())
    return 0;
  
  if(list->getSize()==0)
    return 1;
  
  CharPosItem *item = list->getNextItem();
  CharPosItem *conItem = con->list->getNextItem();
  for( ; item!=0 && conItem!=0; item=list->getNextItem(), conItem=con->list->getNextItem()){
    if(item->getPos()!=conItem->getPos() || item->getChar()!=conItem->getChar())
      return 0;
  }
  return 1;
}

/**
 * Returns the believed backwards anticipation. Hereby, the condition is treated like an effect part.
 */
Perception* Condition::getBackwardsAnticipation(Perception *percept)
{
  Perception *ant = new Perception(percept);
  list->reset();
  CharPosItem *item = list->getNextItem();
  while(item != 0){
    ant->setAttribute(item->getChar(), item->getPos());
    item = list->getNextItem();
  }
  return ant;
}


/**
 * Calculates the number of specified attributes of the condition.
 */
int Condition::getSpecificity()
{
  return list->getSize();
}


ostream& operator<<(ostream& out, Condition *c)
{
  c->list->reset();
  CharPosItem *item=c->list->getNextItem();
  int i;
  for(i=0; i<Perception::length; i++){
    if(item==0 || item->getPos() > i) {
      out <<'#';
    }else{
      out <<item->getChar();
      item=c->list->getNextItem();
    }
  }
  return out;
}
