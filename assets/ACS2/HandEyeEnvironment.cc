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
/     hand-eye gripper environment class
*/

#include<iostream>
#include<fstream>
#include<string.h>

#include"HandEyeEnvironment.h"
#include"Action.h"
#include"Perception.h"


/**
 * Creates a HandEyeEnvironment. Sets all important constants and initializes arrays.
 */
HandEyeEnvironment::HandEyeEnvironment(char *nothing)
{
  envSize = HE_GRID_SIZE*HE_GRID_SIZE+1;
  env = new char[envSize+1];
  env[envSize]='\0';
  originalEnv = new char[envSize+1];
  originalEnv[envSize]='\0';

  blockPositions = new int[2];
  originalBlockPositions = new int[2];

  blockInHand = -1;
  originalBlockInHand = -1;

  gPosX=0;
  gPosY=0;

  originalGPosX=0;
  originalGPosY=0;

  goalGeneratorState=0;
  
  reset();

  cout<<"Read in: "<<endl<<this<< endl;
}

/**
 * Destructor.
 */
HandEyeEnvironment::~HandEyeEnvironment()
{
  delete[] env;
  delete[] originalEnv;
  delete[] blockPositions;  
  delete[] originalBlockPositions;
}

/**
 * Sets the current perception.
 */
void HandEyeEnvironment::getSituation(Perception *perception)
{
  perception->setPerception(env);
}

/**
 * Executes a moving action with all involved consequences.
 */
double HandEyeEnvironment::moveGripper(int xstart, int ystart, int xend, int yend)
{
  if(blockInHand != -1){//block is moved
    blockPositions[blockInHand*2]=xend;
    blockPositions[blockInHand*2+1]=yend;
    env[ystart*HE_GRID_SIZE+xstart]='w';
    env[yend*HE_GRID_SIZE+xend]='b';
  }else{//block is not moved
    env[ystart*HE_GRID_SIZE+xstart]='w';
    env[yend*HE_GRID_SIZE+xend]='g';
  }
  //check if block was under gripper
  if(blockPositions[0]==xstart && blockPositions[1]==ystart)
    env[ystart*HE_GRID_SIZE+xstart]='b';

  //set the feeler last
  env[envSize-1]='0';

  if(blockPositions[0]==xend && blockPositions[1]==yend)
    if( 0!=blockInHand )
      env[envSize-1]='1';
    else if(NOTE_IN_HAND)
      env[envSize-1]='2';
  
  return 0.;
}

/**
 * Executes a gripping action.
 */
double HandEyeEnvironment::gripBlock(int x, int y)
{
  if(blockInHand != -1) //block already in hand
    return 0.;

  if(blockPositions[0]==x && blockPositions[1]==y){
    blockInHand=0;
    env[y*HE_GRID_SIZE+x]='b';
    if(NOTE_IN_HAND)
      env[envSize-1]='2';
    else
      env[envSize-1]='0';
    return 0;//griped block
  }
  return 0.;
}

/**
 * Releases a block if a block is kept in the hand.
 */
double HandEyeEnvironment::releaseBlock(int x, int y)
{
  if(blockInHand != -1){
    env[envSize-1]='1';//set the feeler
    env[y*HE_GRID_SIZE+x]='g';
    blockInHand=-1;
  }
  return 0;
}

/**
 * Executes action 'actt.
 */
double HandEyeEnvironment::executeAction(Action *act)
{
  switch(act->getNr()){
  case 0:
    if(gPosY>0){
      moveGripper(gPosX,gPosY,gPosX,gPosY-1);
      gPosY--;
    }
    break;
  case 1:
    if(gPosX<HE_GRID_SIZE-1){
      moveGripper(gPosX,gPosY,gPosX+1,gPosY);
      gPosX++;
    }
    break;
  case 2:
    if(gPosY<HE_GRID_SIZE-1){
      moveGripper(gPosX,gPosY,gPosX,gPosY+1);
      gPosY++;
    }
    break;
  case 3:
    if(gPosX>0){
      moveGripper(gPosX,gPosY,gPosX-1,gPosY);
      gPosX--;
    }
    break;
  case 4:
    gripBlock(gPosX, gPosY);
    break;
  case 5:
    releaseBlock(gPosX, gPosY);
    break;
  default:
    cout<<"specified action does not exist! action:"<<act->getNr()<<"!?"<<endl;
    break;
  }
  //cout<<"Executed action "<<act<<endl<<this;
  return 0.;
}
  

/**
 * Returns an array of all possible actions in the environment.
 */
Action** HandEyeEnvironment::getActions()
{
  Action **act = new Action*[6];
  act[0]=new Action(0);
  act[1]=new Action(1);
  act[2]=new Action(2);
  act[3]=new Action(3);
  act[4]=new Action(4);
  act[5]=new Action(5);
  return act;
}

/**
 * Converts the specified action to a string. 
 */
char *HandEyeEnvironment::getActionString(Action *act)
{
  char *action=new char[2];
  action[1]='\0';

  switch(act->getNr()){
  case 0:
    action[0]='N'; break;
  case 1:
    action[0]='E'; break;
  case 2:
    action[0]='S'; break;
  case 3:
    action[0]='W'; break;
  case 4:
    action[0]='G'; break;
  case 5:
    action[0]='R'; break;
  default:
    cout<<"specified action code does not exist! action:"<<act->getNr()<<"!?"<<endl;
    break;
  }
  return action;
}

/**
 * Returns the number of actions possible in the environment.
 */
int HandEyeEnvironment::getNoActions()
{
  return 6;
}


/**
 * Returns is the environment should be reset (once a goal is reached).
 * Since no goal is specified in this environment, no reset takes place ever.
 */
int HandEyeEnvironment::isReset()
{
  return 0;
}

/**
 * Resets gripper and block.
 */
int HandEyeEnvironment::reset()
{
  int i;
  for(i=0; i<envSize-1; i++)
    env[i]='w';
  env[i]='0';

  //set the block to random position
  blockPositions[0]=(int)(frand()*HE_GRID_SIZE);
  blockPositions[1]=(int)(frand()*HE_GRID_SIZE);

  env[blockPositions[1]*HE_GRID_SIZE+blockPositions[0]]='b';

  //set the gripper to some position
  if(frand()<0.5){
    //block is in hand
    blockInHand = 0;
    gPosX=blockPositions[0];
    gPosY=blockPositions[1];
    if(NOTE_IN_HAND)
      env[envSize-1]='2';
  }else{
    blockInHand = -1;
    gPosX=(int)(frand()*HE_GRID_SIZE);
    gPosY=(int)(frand()*HE_GRID_SIZE);
    env[gPosY*HE_GRID_SIZE+gPosX]='g';
    if(blockPositions[0]==gPosX && blockPositions[1]==gPosY){
      env[envSize-1]='1';
    }
  }
  return 1;
}

/**
 * Sets environment to test mode.
 */
void HandEyeEnvironment::doTesting()
{
  strcpy(originalEnv,env);

  originalBlockPositions[0]=blockPositions[0];
  originalBlockPositions[1]=blockPositions[1];

  originalBlockInHand = blockInHand;

  originalGPosX=gPosX;
  originalGPosY=gPosY;

  testCounter=0;
}

/**
 * Creates a test in form of a perception action resulting perception triple.
 * Returns if another test was generated.
 */ 
int HandEyeEnvironment::getNextTest(Perception *p0, Action *act, Perception *p1)
{
  if(testCounter>HE_TEST_NO)
    return 0;
  testCounter++;
  reset();
  do{
    getSituation(p0);
    act->setAction((int)(frand()*6));
    executeAction(act);
    getSituation(p1);
  }while( (HE_TEST_ONLY_CHANGES==1 && p0->isEqual(p1)) || (HE_TEST_ONLY_CHANGES==-1 && !p0->isEqual(p1)));
  return 1;
}

/**
 * Resets the environment to normal mode.
 */
void HandEyeEnvironment::endTesting()
{
  strcpy(env, originalEnv);

  blockPositions[0]=originalBlockPositions[0];
  blockPositions[1]=originalBlockPositions[1];

  blockInHand = originalBlockInHand;

  gPosX=originalGPosX;
  gPosY=originalGPosY;
}

/**
 * Creates a goal state. This is the realization of a goal generator in the environment.
 * ACS2 can use the goals to create plans and execute them so that it achieves a complete knowledge faster.
 */
int HandEyeEnvironment::getGoalState(Perception *perception)
{
  if(goalGeneratorState==5){
    goalGeneratorState=0;
    return 0;
  }

  char *goalState = new char[envSize+1];
  strcpy(goalState,env);

  if(blockInHand != -1){
    if(goalGeneratorState==2){
      int x,y;
      do{
	x = (int)(frand()*HE_GRID_SIZE);
	y = (int)(frand()*HE_GRID_SIZE);
      }while(x==gPosX || y==gPosY);
      goalState[gPosY*HE_GRID_SIZE+gPosX]='w';
      goalState[y*HE_GRID_SIZE+x]='b';
      goalGeneratorState=3;
    }else{
      goalState[gPosY*HE_GRID_SIZE+gPosX]='g';
      goalState[envSize-1]='1';
      goalGeneratorState=4;
    }
  }else if(env[envSize-1]=='1'){
    if(goalGeneratorState==4){
      int x,y;
      do{
	x = (int)(frand()*HE_GRID_SIZE);
	y = (int)(frand()*HE_GRID_SIZE);
      }while(x==gPosX || y==gPosY);
      goalState[gPosY*HE_GRID_SIZE+gPosX]='b';
      goalState[y*HE_GRID_SIZE+x]='g';
      goalState[envSize-1]='0';
      goalGeneratorState=5;
    }else{
      if(NOTE_IN_HAND)
	goalState[envSize-1]='2';
      else
	goalState[envSize-1]='0';
      goalState[gPosY*HE_GRID_SIZE+gPosX]='b';
      goalGeneratorState=2;
    }
  }else{
    goalState[gPosY*HE_GRID_SIZE+gPosX]='w';
    goalState[blockPositions[1]*HE_GRID_SIZE+blockPositions[0]]='g';
    goalState[envSize-1]='1';
    goalGeneratorState=1;
  }
  perception->setPerception(goalState);
  delete[] goalState;
  return 1;
}



ostream& operator<<(ostream& out, HandEyeEnvironment *env)
{
  out<<env->envSize<<" = "<<"one block distributed on a "<<HE_GRID_SIZE<<"x"<<HE_GRID_SIZE<<" grid"<<endl;
  if((env->env)[env->envSize-1]=='1')
    out<<"Block under gripper is sensed"<<endl;
  else
    if(env->blockInHand != -1){
      out<<"Gripper transports block"<<endl;
    }else{
      out<<"Gripper is not in contact with a block"<<endl;
    }
  for(int i=0; i<(env->envSize)-1; i++){
    out<<(env->env)[i];
    if((i+1)%HE_GRID_SIZE==0)
      out<<endl;
  }
  return out;
}
