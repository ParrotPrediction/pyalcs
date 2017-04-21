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
/     blocks world environment class
*/

#include<iostream>
#include<fstream>
#include<string.h>

#include"BWEnvironment.h"
#include"Action.h"
#include"Perception.h"


/**
 * Creates a blocks world environment of given number of blocks and stacks.
 */
BWEnvironment::BWEnvironment(char *nothing)
{
  envSize=NR_BLOCKS*NR_STACKS+1;

  env = new char[envSize+1];
  originalEnv = new char[envSize+1];
  
  reset();

  cout<<"Read in: "<<endl<<this<< endl;
}


/**
 * Destructor.
 */
BWEnvironment::~BWEnvironment()
{
  delete[] env;
  delete[] originalEnv;
}


/**
 * Sets the perception to the current perception.
 */
void BWEnvironment::getSituation(Perception *perception)
{
  perception->setPerception(env);
}


/**
 * Executes action 'act' in the blocks world. 
 */
double BWEnvironment::executeAction(Action *act)
{
  int pos = act->getNr()/2;
  int gr = act->getNr()%2;

  if(gr==0){/* gripping */
    if(env[envSize-1]!='.')//gripper has some block already
      return 0;
  }else{
    if(env[envSize-1]=='.')//gripper is empty (cannot release any block)
      return 0;
  }
  // Find first empty position on stack
  int i;
  for(i=0; i<NR_BLOCKS; i++){
    if(env[pos*NR_BLOCKS+i]=='.')
      break;
  }
  if(gr==0){//gripping
    if(i==0)//Nothing to grip
      return 0;
    i--;
    env[envSize-1]=env[pos*NR_BLOCKS+i];
    env[pos*NR_BLOCKS+i]='.';
  }else{//releasing
    env[pos*NR_BLOCKS+i]=env[envSize-1];
    env[envSize-1]='.';
  }
  return 1;
}
  

/**
 * Returns an array of all possible actions in this blocks world.
 */
Action** BWEnvironment::getActions()
{
  Action **act = new Action*[NR_STACKS*2];

  for(int i=0; i< NR_STACKS; i++){
    act[2*i]=new Action(2*i);
    act[2*i+1]=new Action(2*i+1);
  }
  return act;
}

/**
 * Converts an action number to an action string (for output purposes).
 */
char *BWEnvironment::getActionString(Action *act)
{
  char *acode = new char[3];
  acode[2]='\0';
  int nr = act->getNr();
  if(nr>=NR_STACKS*2 || nr<0){
    cerr<<"Error in setAction method: "<<nr<<" is not representable!\n";
    exit(0);
  }
  
  acode[0]=(char)(49+(int)(nr/2));
  acode[1]=(char)(48+(nr%2));
  return acode;
}

/**
 * Returns the number of possible actions in the environment.
 */
int BWEnvironment::getNoActions()
{
  return NR_STACKS*2;
}

/**
 * A reset takes place if all blocks are situated on first stack.
 */
int BWEnvironment::isReset()
{
  int i;
  for(i=0; i<NR_BLOCKS; i++){ /* goal is to put all on first stack! */
    if(env[i]=='.')
      return 0;
  }
  return 1;
}

/**
 * A reset redistributes all blocks randomly.
 */
int BWEnvironment::reset()
{
  int i;
  for(i=0; i<envSize; i++)
    env[i]='.';

  int blockNR[NR_DIFFERENT_BLOCKS];
  for(i=0; i<NR_DIFFERENT_BLOCKS; i++)
    blockNR[i]=0;
  for(i=0; i<NR_BLOCKS; i++)
    blockNR[i%NR_DIFFERENT_BLOCKS]++;

  for(i=0; i<NR_BLOCKS; i++){
    int choose;
    do{
      choose = (int)(frand()*NR_DIFFERENT_BLOCKS);
    }while(blockNR[choose]==0);
    blockNR[choose]--;

    if(i==0 && frand()<0.5){
      env[envSize-1]=(char)((int)'A'+choose);
    }else{
      int position = (int)(frand() * NR_STACKS);

      for(int j=0; j<NR_BLOCKS; j++){
	if(env[position*NR_BLOCKS+j]=='.'){
	  env[position*NR_BLOCKS+j]=(char)((int)'A'+choose);
	  break;
	}
      }
    }
  }
  env[envSize]='\0';
  return 1;
}

/**
 * resetUniform generates a situtation among all possible in the environment uniformly randomly.
 */
void BWEnvironment::resetUniform()
{
  int i;
  for(i=0; i<envSize; i++)
    env[i]='.';

  int blockNR[NR_DIFFERENT_BLOCKS];
  for(i=0; i<NR_DIFFERENT_BLOCKS; i++)
    blockNR[i]=0;
  for(i=0; i<NR_BLOCKS; i++)
    blockNR[i%NR_DIFFERENT_BLOCKS]++;

  int stackArray[NR_BLOCKS+NR_STACKS-1];
  for(i=0; i<NR_BLOCKS+NR_STACKS-1; i++)
    stackArray[i]=-1;

  int blockInGripper=0;
  for(i=0; i<NR_BLOCKS; i++){
    int choose;
    do{
      choose = (int)(frand()*NR_DIFFERENT_BLOCKS);
    }while(blockNR[choose]==0);
    blockNR[choose]--;
    if(i==0 && frand()<0.5){
      env[envSize-1]=(char)((int)'A'+choose);
      blockInGripper=1;
    }else{
      int position=0;
      do{
	position = (int)(frand()*(NR_BLOCKS+NR_STACKS-1-blockInGripper));
      }while(stackArray[position]!=-1);
      stackArray[position]=choose;
    }
  }
  
  int stackPos =0;
  for(i=0; i<NR_BLOCKS+NR_STACKS-1; i++){
    if(stackArray[i]==-1){
      stackPos++;
      if(stackPos>NR_STACKS-1 && i<NR_BLOCKS+NR_STACKS-2){
	cout<<"Overflow Stack!!!"<<endl;
      }
    }else{
      for(int j=0; j<NR_BLOCKS; j++){
	if(env[stackPos*NR_BLOCKS+j]=='.'){
	  env[stackPos*NR_BLOCKS+j]=(char)((int)'A'+stackArray[i]);
	  break;
	}
      }
    }
  }
  env[envSize]='\0';
}

/**
 * Sets the environment to test mode.
 */
void BWEnvironment::doTesting()
{
  //cout<<this;
  strcpy(originalEnv,env);
  testCounter=0;
  reset();
}

/**
 * Specifies a test by specifying a perception action resulting perception triple.
 * Returns 1 if another test was generated.
 */
int BWEnvironment::getNextTest(Perception *p0, Action *act, Perception *p1)
{
  if(testCounter>BW_TEST_NO)
    return 0;
  testCounter++;

  reset();
  
  if(BW_TEST_ALL_POSSIBILITIES_UNIFORMLY){
    resetUniform();
  }

  p0->setPerception(env);
  int randact=0;
  if(BW_TEST_ONLY_CHANGES==1){
    if(env[envSize-1]=='.'){ // gripper is empty -> grip!
      int grippos=0;
      do{
	grippos =(int)(frand()*NR_STACKS);
      }while(env[grippos*NR_BLOCKS]=='.');//Look for a stack with at least one block
      randact = 2*grippos;
    }else{
      randact = (2 * (int)(frand()*NR_STACKS)) +1;
    }
  }else if(BW_TEST_ONLY_CHANGES==-1){
    int change=1;
    do{
      randact = (int)(frand()*NR_STACKS*2);
      if(randact%2==0){//gripping
	if(env[envSize-1]!='.')
	  change=0;
	else if(env[((int)(randact/2))*NR_BLOCKS]=='.')
	  change=0;
      }else{
	if(env[envSize-1]=='.')
	  change=0;
      }
    }while(change);
  }else{
    randact = (int)(frand()*(NR_STACKS*2));
  }

  act->setAction(randact);
  executeAction(act);

  p1->setPerception(env);
  return 1;
}


/**
 * Resets the environment to normal model.
 */
void BWEnvironment::endTesting()//Used for testing purposes
{
  strcpy(env, originalEnv);
}


ostream& operator<<(ostream& out, BWEnvironment *env)
{
  out<<env->envSize<<" = "<<NR_BLOCKS<<" distributed on "<<NR_STACKS<<". Number of different blocks = "<<NR_DIFFERENT_BLOCKS<<endl;

  out <<"Gripper: "<<env->env[env->envSize-1]<<endl;
  for(int y=NR_BLOCKS-1; y>=0; y--){
    for(int x=0; x<NR_STACKS; x++)
      out<<env->env[x*NR_BLOCKS+y];
    out<<endl;
  }
  return out;
}
