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
/     blocks world environment class header
*/

#ifndef _gripper_environment_h_
#define _gripper_environment_h_

#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>

#include"Environment.h"
#include"Perception.h"
#include"Action.h"

#define GRIPPER_REWARD 1000
#define GRIPPER_MAX_WEIGHT 2

class GripperEnvironment: public Environment
{
 public:
  GripperEnvironment(char *nothing);
  ~GripperEnvironment();

  void getSituation(Perception *perception);
  int getPerceptionLength() {return 5;}
  double executeAction(Action *act);
  int isReset();
  int reset();

  int getNoActions();
  Action** getActions();
  char *getActionString(Action *act);

  void doTesting();
  int getNextTest(Perception *p0, Action *act, Perception *p1);
  void endTesting();
  int getGoalState(Perception *perception){return 0;}
  
  char* getID(){char *id=new char[8]; strcpy(id,"Gripper"); return id;}

  friend ostream&
    operator<<(ostream& out, GripperEnvironment *env);

 private:
  char *env;
  int doReset;
};

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0)) 
#endif

#endif

