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
/     mark part class.
*/

#include<iostream>
#include<fstream>

#include"CharCharPosList.h"
#include"CharList.h"
#include"Condition.h"
#include"Perception.h"
#include"PMark.h"

/**
 * Creates a copy of 'mark'
 */
PMark::PMark(PMark *mark)
{
  list = new CharCharPosList(mark->list);
  empty = mark->empty;
}

/** 
 * Specializes the mark in all attributes which are not specified in the conditions, yet.
 * Returns if the mark was actually specialized (enhanced).
 */
int PMark::setMark(Condition *con, Perception *percept)
{
  empty=0;

  if(list->getSize()!=0){
    //if the mark is actually specified already, further specialize all specified attributes
    return setMark(percept);
  }

  int changed=0;
  CharPosList *cpl=con->getList();
  cpl->reset();
  CharPosItem *item=cpl->getNextItem();
  int i;
  for(i=0; i<Perception::length && item!=0; i++){
    for( ; i<item->getPos(); i++){
      changed=1;
      list->insert(percept->getAttribute(i),i);
    }
    item=cpl->getNextItem();
  }
  for( ; i<Perception::length; i++){
    changed=1;
    list->insert(percept->getAttribute(i),i);
  }  
  return changed;
}

/**
 * Directly further specializes all specified attributes in the mark with 'percept'.
 */
int PMark::setMark(Perception *percept)
{
  int changed=0;
  list->reset();
  CharCharPosItem *item;
  for(item= list->getNextItem(); item!=0; item=list->getNextItem())
    changed = item->getItem()->insert(percept->getAttribute(item->getPos())) || changed;
  return changed;
}

/**
 * Determines the strongest differences in between the mark and 'percept'.
 * Returns a condition that specifies all the differences.
 */
Condition* PMark::getDifferences(Perception *percept)
{
  Condition *con=0;
  int nr1=0, nr2=0;
  list->reset();
  CharCharPosItem *item;
  for(item=list->getNextItem(); item!=0; item=list->getNextItem()){
    if(! item->getItem()->doesContain(percept->getAttribute(item->getPos())) )
      nr1++;
    else
      if(item->getItem()->isEnhanced())
	nr2++;
  }

  if(nr1>0){//One or more absolut differences detected -> specialize one radomly chosen one
    con=new Condition();
    nr1 = (int)( frand()* (double)nr1);
    list->reset();
    for(item=list->getNextItem(); item!=0; item=list->getNextItem()){
      if(! item->getItem()->doesContain(percept->getAttribute(item->getPos())) ){
	if(nr1==0){
	  con->specialize(item->getPos(), percept->getAttribute(item->getPos()));
	  break;
	}
	nr1--;
      }
    }
  }else if(nr2 > 0){//One or more equal differences detected -> specialize all of them
    con=new Condition();
    list->reset();
    for(item=list->getNextItem(); item!=0; item=list->getNextItem()) {
      if(item->getItem()->isEnhanced())
	con->specialize(item->getPos(), percept->getAttribute(item->getPos()));
    }
  }else{//Nothing for specialization found!
  }
  return con;
}

/**
 * Returns if the mark is atually empty.
 * Note that a seemingly empty mark can actually contain a mark that is equal 
 * to a completely specified condition. (Therefore the extra parameter 'empty'.
 */
int PMark::isEmpty()
{
  return empty;
}

/**
 * Returns if an attribute in the mark is enhanced.
 */
int PMark::isEnhanced()
{
  list->reset();
  CharCharPosItem *item;
  for(item=list->getNextItem(); item!=0; item=list->getNextItem())
    if(item->getItem()->isEnhanced())
      return 1;
  return 0;
}

/**
 * Checks if mark is equal to other mark. 
 * Note that no entry but not empty is actually the value in the perception
 */
int PMark::isEqual(PMark *m2, Perception *p0)
{
  if(empty && m2->empty)
    return 1;
  
  list->reset();
  m2->list->reset();
  CharCharPosItem *item = list->getNextItem();
  CharCharPosItem *item2 = m2->list->getNextItem();

  while(item!=0 && item2!=0){
    for( ; item != 0 && item->getPos() < item2->getPos(); item = list->getNextItem())
      if( item->getItem()->isEnhanced() || !item->getItem()->doesContain( p0->getAttribute(item->getPos())))
	return 0;
    if(item!=0){
      if( item->getPos() == item2->getPos()){
	if(! item->getItem()->isIdentical(item2->getItem()))
	  return 0;
	item = list->getNextItem();
	item2 = m2->list->getNextItem();
      }else{
	for( ; item2 != 0 && item2->getPos() < item->getPos(); item2 = m2->list->getNextItem())
	  if( item2->getItem()->isEnhanced() || !item2->getItem()->doesContain(p0->getAttribute(item2->getPos())))
	    return 0;
      }
    }
  }
  for( ; item != 0; item = list->getNextItem())
    if( item->getItem()->isEnhanced() || !item->getItem()->doesContain( p0->getAttribute(item->getPos())))
      return 0;
  for( ; item2 != 0; item2 = m2->list->getNextItem())
    if( item2->getItem()->isEnhanced() || !item2->getItem()->doesContain(p0->getAttribute(item2->getPos())))
      return 0;
  return 1;
}

/**
 * Determines whether this mark matches mark m2. 
 * This is the case if all positions that are specified in both are identical.
 */
int PMark::doesMatch(PMark *m2)
{
  if(list->getSize() == 0 || m2->list->getSize() == 0)
    return 1;
  
  list->reset();
  m2->list->reset();
  CharCharPosItem *item = list->getNextItem();
  CharCharPosItem *item2 = m2->list->getNextItem();
  while(item!=0 && item2!=0){
    if(item->getPos() == item2->getPos()){
      if(!item->getItem()->isIdentical(item2->getItem()))
	return 0;
      item = list->getNextItem();
      item2 = m2->list->getNextItem();
    }else{
      for (; item!=0 && item->getPos() < item2->getPos(); item = list->getNextItem()); // go to equal or bigger position
      if(item==0)
	break;
      
      for( ; item2!=0 && item2->getPos() < item->getPos(); item2 = m2->list->getNextItem()); // go to equal or bigger 
    }
  }
  return 1;
}

/**
 * Determines whether this mark matches the perception percept. 
 * This is the case if all attributes in percept are contained in the mark.
 */
int PMark::doesMatch(Perception *p0)
{
  list->reset();
  for(CharCharPosItem *item = list->getNextItem(); item!=0; item = list->getNextItem()){
    if( ! item->getItem()->doesContain(p0->getAttribute(item->getPos())))
      return 0;
  }
  return 1;
}


ostream& operator<<(ostream& out, PMark *pm)
{
  pm->list->reset();
  CharCharPosItem *item=pm->list->getNextItem();
  int i;
  out<<"(";
  for(i=0; i<Perception::length && item!=0; i++, item=pm->list->getNextItem()) {
    for( ; i<item->getPos(); i++)
      out<<"#";
    out<<item->getItem();
  }
  for( ; i<Perception::length; i++)
    out<<"#"; 
  out<<")";
  return out;
}
