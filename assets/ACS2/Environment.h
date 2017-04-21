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
/     base class of all environment classes header
*/


#ifndef _environment_h_
#define _environment_h_

#include"Action.h"
#include"Perception.h"

class Environment
{
 public:
  Environment();
  virtual ~Environment(){;}
  virtual void getSituation(Perception *perception) = 0;
  virtual double executeAction(Action *act) = 0;
  virtual int isReset() = 0;
  virtual int reset() = 0;

  virtual int getPerceptionLength() = 0;
  virtual char *getActionString(Action *act) = 0;
  virtual int getNoActions() = 0;
  virtual Action** getActions() = 0;
  virtual void doTesting() = 0;//Used for testing purposes
  virtual int getNextTest(Perception *p0, Action *act, Perception *p1) = 0;
  virtual void endTesting() = 0;//Used for testing purposes
  virtual int getGoalState(Perception *perception) = 0;
  virtual char *getID()=0;
};

#endif
