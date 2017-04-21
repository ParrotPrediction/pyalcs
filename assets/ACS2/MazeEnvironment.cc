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
/     maze environment class
*/

#include<iostream>
#include<fstream>
#include<string.h>

#include"MazeEnvironment.h"
#include"Action.h"
#include"Perception.h"

/**
 * Constructor of a maze environment. 
 * Reads in the maze specified in file name;
 */
MazeEnvironment::MazeEnvironment(char *fileName)
{
    char s[100], *help;
    int i;

    xSize=0, ySize = 0;
    envSize=8+RANDOMBIT_EXISTS+IRRELEVANT_ATTRIBUTES;

    ifstream *in = new ifstream(fileName, ios::in);

    in->getline(s, 100, '\n');
  
    if(!in || strcmp(s,"")==0){
      cout <<"Please provide correct maze file name!"<<endl;
      exit(0);
    }

    for(; strlen(s)!=0; in->getline(s, 100, '\n')){
	if(xSize==0){
	    xSize=strlen(s);
	    env = new char[xSize+1];
	    strcpy(env,s);
	    env[strlen(s)]='\0';
	}else{
	    if((unsigned int)xSize!=strlen(s)){
		cout<<"Error in File of environment!\n";
		exit(0);
	    }
	    help=env;
	    env = new char[strlen(s) + strlen(help) +1];
	    strncpy(env , help, strlen(help));
	    strcpy(env+strlen(help), s);
	    env[strlen(help) + strlen(s)]='\0';
	    delete[] help;
	}
    }
    delete in;

    ySize=strlen(env)/xSize;

    setFreeRandomPosition(&xPos, &yPos);
  
    doDeterministic=0;

    irrAtt = new char[IRRELEVANT_ATTRIBUTES];
    for(i=0; i<IRRELEVANT_ATTRIBUTES; i++)
      irrAtt[i]=getRandomChar();

    cout<<"Read in: "<<endl<<this<< endl;
}

/**
 * Destructor.
 */
MazeEnvironment::~MazeEnvironment()
{
  delete []env;
  delete []irrAtt;
}

/**
 * Sets the current perception.
 */
void MazeEnvironment::getSituation(Perception *perception)
{
  getSituation(perception, xPos, yPos);
} 

/**
 * Sets to the perception at position xPos, yPos.
 * Random and irrelevant bits are added at the end where applicable.
 */
void MazeEnvironment::getSituation(Perception *perception, int xPos, int yPos)
{
  int i;

  perception->setAttribute(env[((yPos+ySize-1)%ySize)*xSize+xPos], 0);
  perception->setAttribute(env[((yPos+ySize-1)%ySize)*xSize+((xPos+1)%xSize)], 1);
  perception->setAttribute(env[yPos*xSize+((xPos+1)%xSize)], 2);
  perception->setAttribute(env[((yPos+1)%ySize)*xSize+((xPos+1)%xSize)], 3);
  perception->setAttribute(env[((yPos+1)%ySize)*xSize+xPos], 4);
  perception->setAttribute(env[((yPos+1)%ySize)*xSize+((xPos+xSize-1)%xSize)], 5);
  perception->setAttribute(env[yPos*xSize+((xPos+xSize-1)%xSize)], 6);
  perception->setAttribute(env[((yPos+ySize-1)%ySize)*xSize+((xPos+xSize-1)%xSize)], 7);
  
  for(i=0; i<RANDOMBIT_EXISTS;i++)
    perception->setAttribute(getRandomChar(), 8+i);
  
  for(i=0; i<IRRELEVANT_ATTRIBUTES; i++)
    perception->setAttribute(irrAtt[i], 8+RANDOMBIT_EXISTS+i);
}

/**
 * Executes action 'act'.
 * Hereby, the action can lead to an adjacent position if action noise is set.
 */
double MazeEnvironment::executeAction(Action *act)
{
    int dir = act->getNr();;

    if(frand() < SLIPPROB && !doDeterministic){
	if(frand()<0.5)
	    dir = (dir+7)%8;
	else
	    dir = (dir+1)%8;
    }
  
    moveObject(&xPos, &yPos, dir);

    return checkReward(xPos, yPos);
}

/**
 * Creates an array of all possible eight actions in the environment.
 */
Action** MazeEnvironment::getActions()
{
  Action **act = new Action*[8];
  act[0]=new Action(0);
  act[1]=new Action(1);
  act[2]=new Action(2);
  act[3]=new Action(3);
  act[4]=new Action(4);
  act[5]=new Action(5);
  act[6]=new Action(6);
  act[7]=new Action(7);
  return act;
}

/**
 * Converts the coded action to a string.
 */
char *MazeEnvironment::getActionString(Action *act)
{
  char *actString= new char[3];

  switch(act->getNr())
    {
    case 0:
      strcpy(actString,"N"); break;
    case 1:
      strcpy(actString,"NE"); break;
    case 2:
      strcpy(actString,"E"); break;
    case 3:
      strcpy(actString,"SE"); break;
    case 4:
      strcpy(actString,"S"); break;
    case 5:
      strcpy(actString,"SW"); break;
    case 6:
      strcpy(actString,"W"); break;
    case 7:
      strcpy(actString,"NW"); break;
    default:
      cout<<"Error in action string getting"<<endl;
      exit(0);
    }
  return actString;
}

/**
 * Returns the number of actions possible in the environment.
 */
int MazeEnvironment::getNoActions()
{
  return 8;
}

/**
 * Determines if a reset should be applied.
 */
int MazeEnvironment::isReset()
{
    if(env[yPos*xSize + xPos] == FOOD)
	return 1;
    else
	return 0;
}

/**
 * Resets the environment to a random free position.
 */
int MazeEnvironment::reset()
{
    xPos=0;
    yPos=0;
    setFreeRandomPosition(&xPos, &yPos);   
    for(int i=0; i<IRRELEVANT_ATTRIBUTES; i++)
      irrAtt[i]=getRandomChar();    
    return 1;
}

/**
 * Sets environment to test mode (stochastic attributes are disabled).
 */
void MazeEnvironment::doTesting()
{
    doDeterministic=1;
    xTest=0, yTest=0, actTest=-1;
}

/**
 * Creates successively all situation action resulting situation triples that cause a change 
 * as test problems. 
 * Returns if anther test was generated.
 */
int MazeEnvironment::getNextTest(Perception *p0, Action *act, Perception *p1)
{
  int xc, yc;

  while(getNextTestPosition()){
    if(env[yTest*xSize+xTest]==FREE){
      xc=xTest;
      yc=yTest;
      if(moveObject(&xc, &yc, actTest)){
	getSituation(p0, xTest, yTest);
	act->setAction(actTest);
	getSituation(p1, xc, yc);
	return 1;
      }
    }
  }
  return 0;
}

/**
 * Returns the next free position available.
 */
int MazeEnvironment::getNextTestPosition()
{
  actTest++;
  if(actTest >= getNoActions()){
    actTest=0;
    xTest++;
    if(xTest >= xSize){
      xTest=0;
      yTest++;
      if(yTest>=ySize)
	return 0;
    }
  }
  return 1;
}

/**
 * Resets environment to normal mode.
 */
void MazeEnvironment::endTesting()//Used for testing purposes
{
    doDeterministic=0;  
}

/**
 * Returns the ceiling of the double value.
 */
int MazeEnvironment::ceiling(double value)
{
  if((int)value < value){
    return (int)(value+1.);
  }else{
    return (int)value;
  }

}

/**
 * Moves object from position xPos, yPos in direction 'dir'.
 * Returns if a movement took place.
 */
int MazeEnvironment::moveObject(int *xPos, int *yPos, int dir)
{
    int xaim = (((*xPos) + (int) (((ceiling((double)(dir-3)/4.)*2)-1)*(-1)* ceiling((double)(dir%4)/4.)))+xSize)%xSize;
    int yaim = (((*yPos) + (int) (((ceiling( (double)((int)((double)(dir%7)/2.))/4.)*2)-1.)*ceiling((double)((dir+2)%4)/4.)))+ySize) % ySize;
  
    if((env[yaim*xSize + xaim] == FREE || env[yaim*xSize + xaim] == FOOD) 
       && 
       (xaim != this->xPos || yaim != this->yPos || doDeterministic)){
	//Execute movement to free position
	*xPos=xaim;
	*yPos=yaim;
	return 1;
    }
    //Tried to move into a wall or other animat
    return 0;
}

/**
 * Returns reinforcement where appropriate.
 */
double MazeEnvironment::checkReward(int x,int y)
{
    if(env[y*xSize+x]==FOOD)
	return MAZE_REWARD;
    else
	return 0;
}


/**
 * Sets x,y to a free, randomly chosen position.
 */
void MazeEnvironment::setFreeRandomPosition(int *x,int *y)
{
    do{
	(*x)=(int)(frand()*(double)xSize);
	(*y)=(int)(frand()*(double)ySize);
    }while(env[(*y)*xSize + (*x)]!=FREE);
}

/**
 * Returns a randomly 0 or 1
 */
char MazeEnvironment::getRandomChar()
{
    double type;
    char c;

    type=frand()*2;

    switch((int)type){
    case 0:c='0';break;
    case 1:c='1';break;
    default:c='@';break;
    }
    return c;
}



ostream& operator<<(ostream& out, MazeEnvironment *env)
{
  cout<<env->xSize<<" X "<<env->ySize<<"  Maze Environment:"<<endl;
  for(int y=0; y<env->ySize; y++){
    for(int x=0; x<env->xSize; x++){
      if(x==env->xPos && y==env->yPos)
	out << 'A';
      else
	out <<env->env[y*env->xSize+x];
    }
    out <<endl;
  }
  return out;
}
