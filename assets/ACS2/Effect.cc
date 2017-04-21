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
/     effect part class.
*/

#include<iostream>
#include<fstream>

#include"Effect.h"
#include"Perception.h"
#include"ProbCharPosList.h"
#include"ProbCharList.h"
#include"Condition.h"

using namespace std;

/**
 * Constructs a copy of 'eff'.
 */
Effect::Effect(Effect *eff)
{
  list= new ProbCharPosList(eff->list);
}

/**
 * Creates a new enhanced effect part. 
 * Important is that both 'ef1' and 'ef2' are not allowed to be changed since they belong to real classifiers.
 */
Effect::Effect(Effect *ef1, Effect *ef2, double q1, double q2, Perception *percept)
{
  //copy the one list first and then work on the new one and the second effect
  list= new ProbCharPosList(ef1->list);

  list->reset();
  ef2->list->reset();
  
  ProbCharPosItem *item1, *item2;
  item1=list->getNextItem();
  item2=ef2->list->getNextItem();

  while(item1!=0 && item2!=0){
    while(item1 != 0 && item1->getPos() < item2->getPos()){//empty in second means �#�-sign -> merge now with specific attributes!
      item1->getItem()->insert(percept->getAttribute(item1->getPos()), q1, q2);
      item1=list->getNextItem();
    }
    if(item1 == 0)//This happens if the largest position is governed solely by item2
      break;

    if(item1->getPos() == item2->getPos()){//merge two specific attributes
      item1->getItem()->insert(item2->getItem(), q1, q2);
    }

    if(item1->getPos() > item2->getPos()){//empty in first means �#�-sign -> merge now with specific attributes!
      list->insert(percept->getAttribute(item2->getPos()), item2->getPos());
      list->reset();
      for(item1=list->getNextItem(); item1->getPos()!=item2->getPos(); item1=list->getNextItem());//set item1 to current pos
      item1->getItem()->insert(item2->getItem(), q1, q2);
    }
    //At this point item1 and item2 must point to the same position and item1 must be enhanced in this position!
    item1=list->getNextItem();
    item2=ef2->list->getNextItem();
  }
  //Now we still need to do the rest in one of the lists!
  while(item1 != 0) {
    item1->getItem()->insert(percept->getAttribute(item1->getPos()), q1, q2);
    item1=list->getNextItem();    
  }
  while(item2 != 0) {
    //We could also first insert in item2 and then copy the whole thing
    //This would destroy the important structure of item2, though!
    list->insert(percept->getAttribute(item2->getPos()), item2->getPos());
    list->reset();
    for(item1=list->getNextItem(); item1->getPos()!=item2->getPos(); item1=list->getNextItem());//set item1 to current pos
    item1->getItem()->insert(item2->getItem(), q1, q2);    
    item2=ef2->list->getNextItem();
  }  
}

/**
 * Returns the most probable anticipation of the effect part.
 * This is usually the normal anticipation. However, if PEEs are activated, the most probable
 * value of each attribute is taken as the anticipation.
 */
Perception* Effect::getBestAnticipation(Perception *percept)
{
  Perception *ant = new Perception(percept);
  list->reset();
  ProbCharPosItem *item = list->getNextItem();
  while(item != 0){
    ant->setAttribute(item->getItem()->getBestChar(), item->getPos());
    item = list->getNextItem();
  }
  return ant;
}



/**
 * Returns if the effect part anticipates correctly.
 * This is the case if all changes from p0 to p1 are specified and no unchanging attributes are specified.
 * In case of a PEE attribute that contains an unchanging attribute, it is still considered to be correct.
 */
int Effect::doesAnticipateCorrectly(Perception *p0, Perception *p1)
{
  list->reset();
  int i;
  ProbCharPosItem *item;

  for(item=list->getNextItem(), i=0; item!=0; item=list->getNextItem()){
    for(; i<item->getPos(); i++)
      if(p0->getAttribute(i) != p1->getAttribute(i))
	return 0;
    if( ! item->getItem()->doesContain(p1->getAttribute(i)) || ( p0->getAttribute(i) == p1->getAttribute(i) && !item->getItem()->isEnhanced()))
      return 0;
    i++;
  }
  for(; i<Perception::length; i++)
    if(p0->getAttribute(i) != p1->getAttribute(i))
      return 0;
  return 1;
}

/**
 * Returns if the effect part contains PEE attributes.
 */
int Effect::isEnhanced()
{
  list->reset();
  for(ProbCharPosItem *item=list->getNextItem(); item!=0; item=list->getNextItem()){
    if(item->getItem()->isEnhanced())
      return 1;
  }
  return 0;
}

/**
 * Returns if the effect matches the percept.
 * Hereby, the specified attributes are compared with percept.
 * Where the effect part has got #-symbols percept and condPercept are compared. 
 * If they are not equal the effect part does not match.
 */
int Effect::doesMatch(Perception *percept, Perception *condPercept)
{
  list->reset();
  int i=0;
  for(ProbCharPosItem *item=list->getNextItem(); item!=0; item=list->getNextItem(), i++){
    for( ; i<item->getPos(); i++){
      if(percept->getAttribute(i) != condPercept->getAttribute(i))
	return 0;
    }
    if( ! item->getItem()->doesContain(percept->getAttribute(i)))
      return 0;
  }
  for( ; i<Perception::length; i++){
    if(percept->getAttribute(i) != condPercept->getAttribute(i))
      return 0;
  }
  return 1;
}

/**
 * Returns if the effect part specifies at least one of the percepts.
 * An PEE attribute never specifies the corresponding percept.
 */
int Effect::doesSpecifyOnlyChangesBackwards(Perception *backAnt, Perception *situation)
{
  int i=0;
  list->reset();
  for(ProbCharPosItem *item=list->getNextItem(); item!=0; item=list->getNextItem()){
    for( ; i<item->getPos(); i++){
      if(backAnt->getAttribute(i) != situation->getAttribute(i))
	return 0; //change anticipated backwards although no change should occur
    }
    if(item->getItem()->doesContain(backAnt->getAttribute(i)) && !item->getItem()->isEnhanced())
      //if the attribute contains more values, it is not considered to specify one.
      return 0;
    i++;
  }
  for( ; i<Perception::length; i++){
    if(backAnt->getAttribute(i) != situation->getAttribute(i))
      return 0; //change anticipated backwards although no change should occur
  }

  return 1;
}

/**
 * Updates the probabilities of PEE attributes.
 */
void Effect::updateEnhancedEffectProbs(Perception *percept, double updateRate)
{
  list->reset();
  for( ProbCharPosItem *item = list->getNextItem(); item!=0; item = list->getNextItem()){
    item->getItem()->increaseProbability( percept->getAttribute(item->getPos()), updateRate);
  }
}

/**
 * Returns if effect part is similar to effect part 'ef2.
 * In the case of PEE attrbiutes, similarity is the case if no value is only specified by one part.
 */
int Effect::isEqual(Effect *ef2)
{
  list->reset();
  ef2->list->reset();
  
  ProbCharPosItem *item1, *item2;
  item1=list->getNextItem();
  item2=ef2->list->getNextItem();

  while(item1!=0 && item2!=0){
    if( item1->getPos() < item2->getPos() ){//empty in second -> not equal
      return 0;
    }else if( item1->getPos() == item2->getPos() ){//compare two specific anticipations
      if(!item1->getItem()->isSimilar(item2->getItem()))
	return 0;
    }else{//empty in first -> not equal
      return 0;
    }
    item1=list->getNextItem();
    item2=ef2->list->getNextItem();
  }
  //Now we still need to test for possible rest!
  if(item1 != 0)
    return 0;
  if(item2 != 0)
    return 0;
  return 1;
}

/**
 * Specializes the effect part where necessary to correctly anticipate the changes from p0 to p1 and 
 * returns a condition which specifies the attributes which must be specified in the condition part.
 * The specific attributes in the returned conditions are set to the necessary values.
 */
Condition *Effect::getAndSpecialize(Perception *p0, Perception *p1)
{
  Condition *con=new Condition();
  list->reset();
  ProbCharPosItem *item;
  int i;
  for(i=0, item=list->getNextItem(); item!=0; item=list->getNextItem(), i++){
    for( ; i<item->getPos(); i++){
      if(p0->getAttribute(i) != p1->getAttribute(i) ){
	list->insert(p1->getAttribute(i), i);
	con->specialize(i, p0->getAttribute(i));
      }
    }
    if(! item->getItem()->doesContain(p1->getAttribute(i))){
      delete con;
      return 0;
    }
  }
  for( ; i<Perception::length; i++){
    if(p0->getAttribute(i) != p1->getAttribute(i) ){
      list->insert(p1->getAttribute(i), i);
      con->specialize(i, p0->getAttribute(i));
    }
  }
  return con;
}

/**
 * Determines if the effect part is specializable.
 * This is the case if all its currect specific attributes anticipate a change correctly. 
 * If no specific no change is anticipated the classifier is not specializable except 
 * if the attribute is a PEE attribute.
 */
int Effect::isSpecializable(Perception *p0, Perception *p1)
{
  list->reset();
  for(ProbCharPosItem *item=list->getNextItem(); item!=0; item=list->getNextItem()){
    int pos=item->getPos();
    if(!item->getItem()->doesContain(p1->getAttribute(pos)) || (p0->getAttribute(pos)==p1->getAttribute(pos) && !item->getItem()->isEnhanced()))
      return 0;
  }
  return 1;
}



ostream& operator<<(ostream& out, Effect *e)
{
  e->list->reset();
  int i;
  ProbCharPosItem *item;
  for(item=e->list->getNextItem(), i=0; item!=0 && i<Perception::length; item=e->list->getNextItem()){
    for(; i<item->getPos(); i++)
      out<<'#';
    out<<item->getItem();
    i++;
  }
  for(; i<Perception::length; i++)
    out<<'#';
  return out;
}
