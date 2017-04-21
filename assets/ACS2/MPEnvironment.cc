/*
/       (ACS with GA and PEEs in C++)
/	------------------------------------
/	the Anticipatory Classifier System (ACS) with ALP in action set, GA generalization and PEEs
/
/     (c) by Martin Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 11-30-2000
/
/     Multiplexer environment class
*/

#include<iostream>
#include<string.h>
#include<math.h>

#include"MPEnvironment.h"
#include"Action.h"
#include"Perception.h"

/**
 * Constructor of a multiplexer environment. 
 * Calculates the posbits, the reward length and the condition length.
 */
MPEnvironment::MPEnvironment(char *nothing)
{
  for(posbits=1; (int)(1<<posbits)+posbits < MULTIPLEXER; posbits++);/* get posbits */
  
  if(PAYOFFMAP_IN_PERCEPTIONS)
    rcodeLength=posbits+2;
  else
    rcodeLength=1;
  
  condLength = posbits + (int)(1<<posbits) + rcodeLength;
  
  cond = new char[condLength+1];
  testCond = new  char[condLength+1];
  for(int i=0; i<condLength; i++){
    cond[i]='0';
    testCond[i]='0';
  }
  cond[condLength]='\0';
  testCond[condLength]='\0';
  
  doReset=0;
  testNr=0;
  testing=0;
  cout<<"Constructed Multiplexer with CL:"<<condLength<<" RCL:"<<rcodeLength<<" PB:"<<posbits<<endl;
}

/**
 * Destructor.
 */
MPEnvironment::~MPEnvironment()
{
  delete[] cond;
  delete[] testCond;
}

/**
 * Sets the current perception.
 */
void MPEnvironment::getSituation(Perception *perception)
{
  perception->setPerception(cond);
}

/**
 * Executes action A.
 * In the multiplexer problem the result is the coding of the correctness 
 * (reward/reward map) in the perceptions.
 * Returns reinforcement 
 */
double MPEnvironment::executeAction(Action *A)
{
  int reward=0 , rewardRet=0, rewardMapRet=0;
  
  int correct=1;

  int i,place;
  for(i=0, place=posbits; i<posbits; i++){
    if(cond[i]=='1')
      place += (int)(1<<(posbits-1-i));
  }
  if(A->getNr() == (int)(cond[place]-'0')){
    reward= 300+(((place-posbits)*200)+100*(int)(cond[place]-'0'));
    rewardRet=1000;
  }else{
    correct=0;
    reward = (((place-posbits)*200)+100*(int)((int)cond[place]-(int)'0'));
    rewardRet=0;
  }
  rewardMapRet += reward;

  if(PAYOFFMAP_IN_PERCEPTIONS) {
    /*Calculate the place of the reward attributes*/
    place= condLength - rcodeLength;
    int rewardnr=reward/100;
    /*set the reward - corresponding attributes*/
    for( ; place<condLength; place++){
      if( (int)(1<<(condLength-place-1)) <= rewardnr){
	cond[place]='1';
	rewardnr-= (int)(1<<(condLength-place-1));
      }
    }
  } else {
    cond[condLength-1]= '0'+(correct + WHICHCODE0112);
  }
  doReset=1;
  
  if(RETURN_REWARDMAP)
    return (double)rewardMapRet;
  return (double)rewardRet;
}

/**
 * Returns an array of all possible actions.
 */
Action** MPEnvironment::getActions()
{
  Action **act = new Action*[2];

  for(int i=0; i<2; i++)
    act[i]=new Action(i);

  return act;
}

/**
 * Converts the action to a string.
 */
char* MPEnvironment::getActionString(Action *act)
{
  char *action = new char[2];
  if(act->getNr()==0)
    action[0]='0';
  else
    action[0]='1';
  action[1]='\0';
  return action;
}

/**
 * Returns the number of actions possible in this environment.
 */
int MPEnvironment::getNoActions()
{
  return 2;
}

/**
 * Returns if a reset should take place.
 * In the MP environment thsi is always the case afer the execution of any action.
 */
int MPEnvironment::isReset()
{
  return doReset;
}

/**
 * Creates the next problem instance.
 */
int MPEnvironment::reset()
{
  /*Set a new random state*/
  int i;
  for(i=0; i<condLength-rcodeLength; i++)
    cond[i]=(char)(frand()*2)+'0';

  /*Set the reward part of the string to '0's*/
  for(; i<condLength; i++)
    cond[i]='0';

  doReset=0;
  return 1;
}

/**
 * Sets to test mode.
 */ 
void MPEnvironment::doTesting()//Used for testing purposes
{
  testNr=0;
  testing=1;
  strcpy(testCond, cond);
}

/**
 * Resets to normal mode.
 */
void MPEnvironment::endTesting()//Used for testing purposes
{
  testing=0;
  strcpy(cond, testCond);
}

/**
 * Creates next test problem instance.
 */
int MPEnvironment::getNextTest(Perception *p0, Action *act, Perception *p1)
{
  if(testNr < MP_TEST_SIZE){
    reset();
    p0->setPerception(cond);
    act->setAction((int)(frand()*getNoActions()));
    executeAction(act);
    p1->setPerception(cond);
    testNr++;
    return 1;
  }
  return 0;
}

ostream& operator<<(ostream& out, MPEnvironment *env)
{
  out <<"This is a multiplexer environment with"<<endl;
  out <<env->posbits<<" position bits, "<<env->condLength<<" condition length, and "<<env->rcodeLength<<" reward code length!"<<endl;
  return out;
}
