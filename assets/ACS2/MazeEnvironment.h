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
/     maze environment class header
*/


#ifndef _maze_environment_h_
#define _maze_environment_h_

#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#include"Environment.h"
#include"Perception.h"
#include"Action.h"

#define FOOD 'F'
#define FREE '*'
#define OBSTACLE 'O'

#define IRRELEVANT_ATTRIBUTES 0
#define RANDOMBIT_EXISTS 0
#define SLIPPROB 0.00

#define MAZE_REWARD 1000

class MazeEnvironment: public Environment
{
 public:
  MazeEnvironment(char *fileName);
  ~MazeEnvironment();

  void getSituation(Perception *perception);
  double executeAction(Action *act);
  int isReset();
  int reset();
  
  int getNoActions();
  Action** getActions();
  char *getActionString(Action *act);

  void doTesting();//Used for testing purposes
  int getNextTest(Perception *p0, Action *act, Perception *p1);
  void endTesting();//Used for testing purposes
  int getGoalState(Perception *perception){return 0;}
  int getPerceptionLength() {return envSize;}

  char* getID(){char *id=new char[5]; strcpy(id,"Maze"); return id;}
  
  friend ostream&
    operator<<(ostream& out, MazeEnvironment *env);

 private:
  void getSituation(Perception *perception, int xPos, int yPos);
  int getNextTestPosition();
  int ceiling(double value);
  int moveObject(int *xPos, int *yPos, int dir);
  double checkReward(int x,int y);
  void setFreeRandomPosition(int *x,int *y);
  char getRandomChar();
  
  int xPos, yPos;
  int xSize, ySize;
  int xTest, yTest, actTest;
  char *env;
  
  int envSize;
  int doDeterministic;
  char *irrAtt;
};

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0)) 
#endif

#endif
