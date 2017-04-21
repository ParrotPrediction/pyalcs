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
/     classifier list class header
*/

#ifndef _classifier_list_h_
#define _classifier_list_h_


#include<iostream>
#include<fstream>
#include"Classifier.h"
#include"Action.h"
#include"ACSConstants.h"
#include"Environment.h"

using namespace std;

class PureClassifierList
{
  friend class ClassifierList;
  friend ostream& operator<<(ostream& out, PureClassifierList *pcll);
 private:
  PureClassifierList() {cl=0; next=0;}
  PureClassifierList(Classifier *cl) {this->cl = cl; next=0;}
  PureClassifierList(PureClassifierList *list);
  ~PureClassifierList() {}

  Classifier *cl;
  PureClassifierList *next;
};

class ClassifierList
{
 public:
  ClassifierList(Environment *env);
  ClassifierList(ClassifierList *cllist);
  ClassifierList(ClassifierList *clList, Classifier *cl);
  ClassifierList(ClassifierList *cllist, Perception *percept);
  ClassifierList(Perception *percept, ClassifierList *pop);
  ClassifierList(ClassifierList *cllist, Action *act);
  ClassifierList(ClassifierList *clList, double quality);
  ~ClassifierList();
  
  void deleteClassifiers();
  void addClassifier(Classifier *cl);
  int deleteClassifierPointer(Classifier *clp);
  void applyALP(Perception *p0, Action *act, Perception *p1, int time, ClassifierList *pop, ClassifierList *matchSet);
  void applyGA(int time, ClassifierList *pop, ClassifierList *matchSet, Perception *p1);
  void applyReinforcementLearning(double rho, double P);
  void doOneStepMentalActing(int steps);
  void setGATimeStamps(int time);
  void setALPTimeStamps(int time);
  void chooseAction(Action *act, ClassifierList *pop, Perception *situation);
  void chooseBestQRAction(Action *act);

  Classifier *getSimilar(Classifier *cl);
  Classifier *getSubsumer(Classifier *cl);

  int existClassifier(Perception *s1, Action *act, Perception *s2, double quality);

  void deleteClassifier(Classifier *cl);
  void checkList();

  double getMaximumQR();
  Classifier *getHighestQualityClassifier();
  int getSize();
  int getNumSize();
  double getSpecificity();

  Action** searchGoalSequence(Perception *start, Perception *goal);
  Action **searchOneForwardStep(ClassifierList *relList, int *fs, int fSize, int fPoint, Perception **arrayPF, 
				int bSize, Perception **arrayPB, ClassifierList **arrayListF, ClassifierList **arrayListB);
  Action **searchOneBackwardStep(ClassifierList *relList, int *bs, int bSize, int bPoint, Perception **arrayPB, 
				 int fSize, Perception **arrayPF, ClassifierList **arrayListB, ClassifierList **arrayListF);

  friend ostream&
    operator<<(ostream& out, ClassifierList *cll);

 private:
  ClassifierList *uselessCase(Perception *p0, ClassifierList *population, ClassifierList *matchSet, int time);

  void addMatchingClassifiers(ClassifierList *newSet, Perception *percept);
  void addMatchingClassifier(Classifier *cl, Perception *p0);

  void addClassifierList(ClassifierList *cllist);

  ClassifierList() {size=0; list=0; env=0;}
  void deleteGAClassifiers(ClassifierList *pop, ClassifierList *set, int childNo);
  void chooseExploreAction(Action *act);
  void chooseBestAppAvAction(Action *act);
  void chooseBestKnowArrayAction(Action *act);
  void chooseRandomAction(Action *act);
  void chooseLatestAction(Action *act);
  void chooseLookaheadAction(Action *act, ClassifierList *pop, Perception *situation);
  
  int doApplyGA(int time);
  void selectParents(Classifier **parent1, Classifier **parent2);
  
  void applyEnhancedEffectPartCheck(ClassifierList **newList, Perception *p0, int time);

  int insertALPOffspringToNewList(Classifier *child, ClassifierList **newList);
  int checkGASimilarities(Classifier *child);
  
  int doesContainState(Perception **ap, int asize, Perception *state);
  void freeBidisearchStuff(ClassifierList **al1, Perception **ap1, int size1, ClassifierList **al2, Perception **ap2, int size2, ClassifierList *cllist);


  Environment *env;
  PureClassifierList *list;
  int size;
};

#endif
