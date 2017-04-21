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
/     collwill and rescorla's rat experiments
*/

#include<iostream>
#include<fstream>
#include<string.h>

#include"CRRatEnvironment.h"
#include"Action.h"
#include"Perception.h"

/**
 * Constructor of a maze environment. 
 * Reads in the maze specified in file name;
 */
CRRatEnvironment::CRRatEnvironment(char *nothing)
{
  timer=0;
  phase=0;
  doReset=0;

  if(CR_DO90==0){
    situation = new char[5];
    length=4;
  }else{
    situation = new char[7];
    length=6;
  }
  int i;
  for(i=0; i< length; i++)
    situation[i]='0';
  situation[i]='\0';

  cout<<"Read in: "<<endl<<this<< endl;
}

/**
 * Destructor.
 */
CRRatEnvironment::~CRRatEnvironment()
{
  delete[] situation;
}

/**
 * Sets the current perception.
 */
void CRRatEnvironment::getSituation(Perception *perception)
{
  perception->setPerception(situation);
} 


/**
 * Executes action 'act'.
 */
double CRRatEnvironment::executeAction(Action *act)
{
  int a = act->getNr();;
  
  //handle testing phase first (always reset, never provide effects)
  if((!CR_DO90 && phase==2) || (CR_DO90 && phase==3)){
    if(a==0 || a==1){
      doReset=1;
      if(!CR_DO90){
	if(a==1)
	  return 1;
      }else{
	if((a==0 && situation[5]=='1') || (a==1 && situation[4]=='1'))
	  return 1;
      }
    }
    return 0;
  }

  if(a==3)//do nothing action
    return 0.;

  if(a==2){//consumption action
    if((CR_DO90==0 && phase==1) || (CR_DO90 && phase==2)){
      if(situation[0]=='1'){
	situation[0]='0';
	doReset=1;
	return 0;
      }else{
	if(situation[1]=='1'){
	  situation[1]='0';
	  doReset=1;
	  return 1000;
	}
      }
    }else{
      if(situation[0]=='1' || situation[1]=='1'){
	situation[0]='0';
	situation[1]='0';
	doReset=1;
	return 1000;
      }
    }
  }
  if(a==0 || a==1){
    if(situation[0]=='1' || situation[1]=='1')
      return 0; //if food is available no additional reinforcers are provided
    if(a==0){//try to press
      if(situation[2]=='1'){//lever is available
	if(CR_DO90==0 && phase==0){
	  situation[0]='1';
	}else if(CR_DO90 && phase!=3){
	  if(situation[4]=='1'){
	    situation[0]='1';
	  }else if(situation[5]=='1'){
	    situation[1]='1';
	  }
	}
      }
    }else{ //try to pull
      if(situation[3]=='1'){//chain available
	if(CR_DO90==0 && phase==0){
	  situation[1]='1';
	}else if(CR_DO90 && phase!=3){
	  if(situation[4]=='1')
	    situation[1]='1';
	  else if(situation[5]=='1')
	    situation[0]='1';
	}
      }
    }
  }
  return 0;
}

/**
 * Creates an array of all possible eight actions in the environment.
 */
Action** CRRatEnvironment::getActions()
{
  Action **act = new Action*[4];
  act[0]=new Action(0);
  act[1]=new Action(1);
  act[2]=new Action(2);
  act[3]=new Action(3);
  return act;
}

/**
 * Converts the coded action to a string.
 */
char *CRRatEnvironment::getActionString(Action *act)
{
  char *actString= new char[3];

  switch(act->getNr())
    {
    case 0:
      strcpy(actString,"Pr"); break;
    case 1:
      strcpy(actString,"Pu"); break;
    case 2:
      strcpy(actString,"Ea"); break;
    case 3:
      strcpy(actString,"No"); break;
    default:
      cout<<"Error in action string getting"<<endl;
      exit(0);
    }
  return actString;
}

/**
 * Returns the number of actions possible in the environment.
 */
int CRRatEnvironment::getNoActions()
{
  return 4;
}

/**
 * Determines if a reset should be applied.
 */
int CRRatEnvironment::isReset()
{
  return doReset;
}


/**
 * Resets the environment to a random free position.
 */
int CRRatEnvironment::reset()
{
  doReset=0;
  if(!CR_DO90) {
    if(phase==0){
      situation[2]='0';
      situation[3]='0';
      if(timer==CR85_PHASE1TIME){
	timer=0;
	phase=1;
      }else{
	if(frand()<0.5)
	  situation[2]='1';
	else
	  situation[3]='1';
      }
    }
    if(phase==1){
      if(timer==CR85_PHASE2TIME){
	timer=0;
	phase=2;
      }else{
	if(frand()<0.5)
	  situation[0]='1';
	else
	  situation[1]='1';
      }
    }
    if(phase==2){
      if(timer==CR85_PHASE3TIME){
	timer=0;
	phase=0;
	return 0;
      }else{
	situation[2]='1';
	situation[3]='1';
      }
      timer++;
      return 2;
    }
  }else{//experiment CR 90
    if(phase==0 || phase==1){
      situation[2]='0';
      situation[3]='0';
      situation[4]='0';
      situation[5]='0';
      if(phase==0 && timer==CR90_PHASE1TIME){
	timer=0;
	phase=1;
      }
      if(phase==1 && timer==CR90_PHASE2TIME){
	timer=0;
	phase=2;
      }else{
	if(phase==0){
	  if(frand()<0.5){
	    situation[2]='1';
	  }else{
	    situation[3]='1';
	  }
	}else if(phase==1){
	  situation[2]='1';
	  situation[3]='1';
	}
	if(frand()<0.5)
	  situation[4]='1';
	else
	  situation[5]='1';
      }
    }
  }
  if(phase==2) {
    if(timer==CR90_PHASE3TIME){
      timer=0;
      phase=3;
    }else{
      if(frand()<0.5)
	situation[0]='1';
      else
	situation[1]='1';
    }
  }
  if(phase==3){
    if(timer==CR90_PHASE4TIME){
      timer=0;
      phase=0;
      return 0;
    }else{
      situation[2]='1';
      situation[3]='1';
      situation[4]='0';
      situation[5]='0';
      if(frand()<0.5)
	situation[4]='1';
      else
	situation[5]='1';
    }
    timer++;
    return 2;
  }
  timer++;
  return 1;
}

/**
 * Sets environment to test mode (stochastic attributes are disabled).
 */
void CRRatEnvironment::doTesting()
{
}

/**
 * Creates successively all situation action resulting situation triples that cause a change 
 * as test problems. 
 * Returns if anther test was generated.
 */
int CRRatEnvironment::getNextTest(Perception *p0, Action *act, Perception *p1)
{
  return 0;
}

/**
 * Resets environment to normal mode.
 */
void CRRatEnvironment::endTesting()//Used for testing purposes
{
}


ostream& operator<<(ostream& out, CRRatEnvironment *env)
{
  out<<env->situation<<endl;
  
  return out;
}
