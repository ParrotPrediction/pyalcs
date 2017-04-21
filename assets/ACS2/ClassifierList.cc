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
/     classifier list class.
*/

#include<iostream>
#include<fstream>
#include<assert.h>
#include"ClassifierList.h"
#include"Classifier.h"
#include"Action.h"
#include"ACSConstants.h"
#include"Environment.h"

/**
 * Constructor that copies whole list - does not copy the classifiers!
 */
PureClassifierList::PureClassifierList(PureClassifierList *list)
{
  cl = list->cl; 
  if(list->next!=0)
    assert((next = new PureClassifierList(list->next))!=0);
  else
    next=0;
}

/**
 * Destructor deletes all classifier pointer of the PureClassifierList.
 * Assures a small stack.
 */
ClassifierList::~ClassifierList()
{
  PureClassifierList *pcl=list, *pcln;
  while(pcl!=0){
    pcln=pcl->next;
    delete pcl;
    pcl=pcln;
  }
}

/**
 * Looks for identical classifiers and possible empty parts of the classifier (O(n^2) !)
 * This method is only used for bug checks.
 */
void ClassifierList::checkList()
{
  PureClassifierList *listp;
  for(listp=list; listp!=0; listp=listp->next){
    if(listp->cl->getCondition()==0 || listp->cl->getAction()==0 || listp->cl->getEffect()==0 || listp->cl->getPMark()==0 || listp->cl->getEffect()->isEnhanced()){
      cout<<this;
      cout<<"Detected Error in classifier List!";
      cout<<listp->cl;
      exit(0);
    }
  }
  for(listp=list; listp!=0; listp=listp->next){
    for(PureClassifierList *lp2=listp->next; lp2!=0; lp2=lp2->next){
      if(listp->cl->isSimilar(lp2->cl)){
	cout<<listp->cl<<endl<<lp2->cl<<endl;
	assert(1==0);
      }
    }
  }
}

/**
 * Constructor meant for initialization the population of ACS2.
 * Due to the not anymore existing special status of the most general classifiers, 
 * it is actually not necessary to add classifiers in the beginning.
 */
ClassifierList::ClassifierList(Environment *env)
{
  list=0;
  size=0;
  this->env=env;
}

/**
 * Copy Constructor of list. (Does not copy classifiers.)
 */
ClassifierList::ClassifierList(ClassifierList *cllist)
{
  if(cllist->list==0)
    list=0;
  else
    list = new PureClassifierList(cllist->list);
  size=cllist->size;
  env=cllist->env;
}

/**
 * Construct classifier list out of list with q_{cl}>quality. (Does not copy classifiers.)
 */
ClassifierList::ClassifierList(ClassifierList *clList, double quality)
{
  list = 0;
  size=0;
  env = clList->env;
  for(PureClassifierList *listp=clList->list; listp!=0; listp=listp->next){
    if(listp->cl->getQuality()>quality){
      addClassifier(listp->cl);
    }
  }
}

/**
 * Construct classifier list out of clList with all classifiers that classifier cl 'links'.
 */
ClassifierList::ClassifierList(ClassifierList *clList, Classifier *cl)
{
  list = 0;
  size=0;
  env = clList->env;
  for(PureClassifierList *listp=clList->list; listp!=0; listp=listp->next){
    if(cl->doesLink(listp->cl)){
      addClassifier(listp->cl);
    }
  }
}


/**
 * Constructor for match set generation. (Does not copy classifiers.)
 */
ClassifierList::ClassifierList(ClassifierList *pop, Perception *percept)
{
  list=0;
  size=0;
  env = pop->env;
  for(PureClassifierList *listp=pop->list ; listp!=0; listp=listp->next){
    if(listp->cl->doesMatch(percept)){
      addClassifier(listp->cl);
    }
  }
}

/**
 * Constructor for backwards match set generation.
 * This method is used in the backwards search procedure. 
 * Anticipations and the specified conditions need to match...
 * (Does not copy classifiers.)
 */
ClassifierList::ClassifierList(Perception *percept, ClassifierList *pop)
{
  list=0;
  size=0;
  env=pop->env;
  for(PureClassifierList *listp=pop->list; listp!=0; listp=listp->next){
    if(listp->cl->doesMatchBackwards(percept)){
      addClassifier(listp->cl);
    }
  }
}

/**
 * Constructor for action set generation
 *  (Does not copy classifiers.)
 */
ClassifierList::ClassifierList(ClassifierList *matchSet, Action *act)
{
  list=0;
  size=0;
  env = matchSet->env;

  for(PureClassifierList *listp=matchSet->list ; listp!=0; listp=listp->next){
    if(listp->cl->hasAction(act)){
      addClassifier(listp->cl);
    }
  }
}

/**
 * Adds all in percept matching classifiers from the new set newSet to the current match set.
 * This method is privat and ClassifierList assures that only distinct classifiers are in newSet.
 */
void ClassifierList::addMatchingClassifiers(ClassifierList *newSet, Perception *percept)
{
  if(newSet!=0){
    for(PureClassifierList *listp=newSet->list; listp!=0; listp=listp->next){
      addMatchingClassifier(listp->cl, percept);
    }
  }
}

/**
 * Checks if classifier cl matches percept. 
 * If so, classifier is added to the current list
 * This method is privat and ClassifierList assures that cl is always 
 * a from the set distinct classifier.
 */
void ClassifierList::addMatchingClassifier(Classifier *cl, Perception *percept)
{
  if(cl->doesMatch(percept)){
    //add the matching guy in front
    PureClassifierList *lp = new PureClassifierList(cl);
    lp->next = list;
    list = lp;
    size++;
  }
}

/**
 * Returns 1 if there is a classifier in this list with a quality q higher than quality 
 * that maches s1, specifies act, and predicts s2. Returns 0 otherwise.
 */
int ClassifierList::existClassifier(Perception *p1, Action *act, Perception *p2, double quality)
{
  for(PureClassifierList *listp=list; listp!=0; listp=listp->next){
    if(listp->cl->getQuality() > quality)
      if(listp->cl->doesMatch(p1))
	if(listp->cl->hasAction(act))
	  if(listp->cl->doesAnticipateCorrect(p1, p2))
	    return 1;
  }
  return 0;
}

/**
 * Adds classifier list clList to the current list.
 * The method is privat and ClassifierList assures that each classifier in 
 * clList is not yet contained in the current list.
 */
void ClassifierList::addClassifierList(ClassifierList *clList)
{
  if(clList==0)
    return;

  PureClassifierList *listp, *help;
  for(listp=clList->list ; listp!=0; listp=listp->next){
    help = new PureClassifierList(listp->cl);
    help->next = list;
    list=help;
  }
  size += clList->size;
}

/**
 * Returns the maximum q_{cl}*r_{cl} value among those classifier in the list that anticipte 
 * a change in the environment.
 */
double ClassifierList::getMaximumQR()
{
  double max=0;
  int found=0;
  if(list!=0){
    for(PureClassifierList *listp = list; listp!=0; listp=listp->next){
      if(listp->cl->doesAnticipateChange())
	if(!found || max < listp->cl->getQuality() * listp->cl->getRewardPrediction()){
	  max = listp->cl->getQuality() * listp->cl->getRewardPrediction();
	  found=1;
	}
    }
  }
  return max;
}

/**
 * Returns the classifier with the highest quality.
 */
Classifier *ClassifierList::getHighestQualityClassifier()
{
  double qMax=0.;
  Classifier *clMax=0;
  for(PureClassifierList *listp=list; listp!=0; listp=listp->next){
    if(listp->cl->getQuality() > qMax){
      qMax=listp->cl->getQuality();
      clMax = listp->cl;
    }
  }
  return clMax;
}

/**
 * Chooses action according to epsilon greedy policy.
 */
void ClassifierList::chooseAction(Action *act, ClassifierList *pop, Perception *situation)
{
  if(frand()<EPSILON){
    chooseExploreAction(act);
  }else{
    if(DO_LOOKAHEAD_WINNER)
      chooseLookaheadAction(act, pop, situation);
    else
      chooseBestQRAction(act);
  }
}

/**
 * Chooses action according to current exploration policy
 */
void ClassifierList::chooseExploreAction(Action *act)
{
  if(frand() < PROB_EXPLORATION_BIAS)
    if(EXPLORATION_BIAS_METHOD==0 || (EXPLORATION_BIAS_METHOD==2 && frand() < 0.5) )
      chooseLatestAction(act);
    else
      chooseBestKnowArrayAction(act);
  else
    chooseRandomAction(act);
}

/**
 * Chooses action which has been executed most long ago (as it seems)
 */
void ClassifierList::chooseLatestAction(Action *act)
{
  int lAction=0;
  double lValue=-1.;
  const int noActions = env->getNoActions();
  int *noValues= new int[noActions];
  int i;
  for(i=0; i<noActions; i++)
    noValues[i]=0;
  
  PureClassifierList *listp = list;
  if(listp!=0){
    lAction=listp->cl->getAction()->getNr();
    lValue=listp->cl->getALPTimeStamp();
    noValues[lAction] += listp->cl->getNumerosity();
    
    for(listp = list->next; listp!=0; listp=listp->next){
      noValues[listp->cl->getAction()->getNr()] += listp->cl->getNumerosity();
      if(listp->cl->getALPTimeStamp() < lValue){
	lAction = listp->cl->getAction()->getNr();
	lValue = listp->cl->getALPTimeStamp();
      }
    }
  }
  /* If no classifier represents an action, choose it for execution! */
  for(i=0; i<noActions; i++){
    if(noValues[i]==0)
      lAction=i;
  }
  act->setAction(lAction);
  delete[] noValues;
}

/**
 * Chooses the action with the qualtiy weighted average application average value 
 * of all micro-classifiers in the current list. Unrepresented actions are always chosen for execution.
 */
void ClassifierList::chooseBestAppAvAction(Action *act)
{
  const int noActions = env->getNoActions();
  double *delayValues = new double[noActions];
  double *noDelayValues = new double[noActions];

  //cout<<this;
  for(int i=0; i<noActions; i++){
    delayValues[i] = 0.;
    noDelayValues[i] = 0;
  }

  for(PureClassifierList *listp = list; listp!=0; listp=listp->next){
      delayValues[listp->cl->getAction()->getNr()] += listp->cl->getApplicationAverage() * listp->cl->getQuality()* listp->cl->getNumerosity();
      noDelayValues[listp->cl->getAction()->getNr()] += listp->cl->getQuality()* listp->cl->getNumerosity();
  }

  int ldAction=0;
  double ldValue=-1.;
  for(int j=0; j<noActions; j++){
    if(noDelayValues[j]==0){
      //always choose unrepresented actions!
      ldAction=j;
      break;
    }

    if(delayValues[j]/noDelayValues[j] > ldValue) {
      ldAction=j;
      ldValue = delayValues[j]/noDelayValues[j];
    }
  }
  //cout<<endl;
  act->setAction(ldAction);
  delete[] delayValues;
  delete[] noDelayValues;
}

/**
 * Creates 'knowledge array' that represents the average quality of the anticipation for each action
 * in the current list. Chosen is the action, ACS2 knows least about the consequences.
 */
void ClassifierList::chooseBestKnowArrayAction(Action *act)
{
  const int noActions = env->getNoActions();
  double *knowValues = new double[noActions];
  double *noKnowValues = new double[noActions];

  for(int i=0; i<noActions; i++){
    knowValues[i] = 0.;
    noKnowValues[i] = 0;
  }

  for(PureClassifierList *listp = list; listp!=0; listp=listp->next){
    knowValues[listp->cl->getAction()->getNr()] += listp->cl->getQuality()* listp->cl->getNumerosity();
    noKnowValues[listp->cl->getAction()->getNr()] += listp->cl->getNumerosity();
  }

  int lkAction=0;
  double lkValue=2.;
  for(int j=0; j<noActions; j++){
    if(noKnowValues[j]==0 ){
      //choose unknow action first
      lkAction=j;
      lkValue=0;
    }else if(knowValues[j]/noKnowValues[j] < lkValue) {
      lkAction=j;
      lkValue = knowValues[j]/noKnowValues[j];
    }
  }
  act->setAction(lkAction);
  delete[] knowValues;
  delete[] noKnowValues;
}

/**
 * Chooses an action considering a one step anticipation - the lookahead winner algorithm!
 * Note: epsilon is considered in this function!
 */
void ClassifierList::chooseLookaheadAction(Action *act, ClassifierList *pop, Perception *situation)
{
  const int noActions = env->getNoActions();
  Classifier **bestKnowClassifier = new Classifier*[noActions];
  int i, actNr;
  for(i=0; i<noActions; i++){
    bestKnowClassifier[i] = 0;
  }
  PureClassifierList *listp; 
  for(listp = list; listp!=0; listp=listp->next){
    if(listp->cl->doesAnticipateChange()) {
      actNr = listp->cl->getAction()->getNr();
      if(bestKnowClassifier[actNr]==0 || bestKnowClassifier[actNr]->getQuality()<listp->cl->getQuality())
	bestKnowClassifier[actNr]=listp->cl;
    }
  }
  
  double *bestActValues = new double[noActions];
  for(i=0; i<noActions; i++){
    if(bestKnowClassifier[i]!=0){
      Perception *ant = bestKnowClassifier[i]->getBestAnticipation(situation);
      ClassifierList *matchSet = new ClassifierList(pop, ant);
      bestActValues[i] = matchSet->getMaximumQR();
      delete matchSet;
      delete ant;
    }else{
      bestActValues[i]=0;
    }
  }
  
  double max=-100000;
  Classifier *maxCl=0;
  
  for(listp = list; listp!=0; listp=listp->next){
    actNr = listp->cl->getAction()->getNr();
    if( bestKnowClassifier[actNr]==0 && max < listp->cl->getQuality() * listp->cl->getRewardPrediction()){
      max = listp->cl->getQuality() * listp->cl->getRewardPrediction();
      maxCl = listp->cl;
    }else if( bestKnowClassifier[actNr]!=0 && max < (listp->cl->getQuality() * listp->cl->getRewardPrediction() + bestKnowClassifier[actNr]->getQuality() * GAMMA * bestActValues[actNr]) / (1 + bestKnowClassifier[actNr]->getQuality() * GAMMA)) {
      max = (listp->cl->getQuality() * listp->cl->getRewardPrediction() + bestKnowClassifier[actNr]->getQuality() * GAMMA * bestActValues[actNr]) / (1 + bestKnowClassifier[actNr]->getQuality() * GAMMA);
      maxCl = listp->cl;      
    }
  }
  
  if(maxCl!=0)
    act->setAction(maxCl->getAction());
  else
    chooseRandomAction(act);
  delete[] bestKnowClassifier;
  delete[] bestActValues;
}

/**
 * Chooses best action according to q*r. 
 * Requires that chosen classifier anticipates a change in the environment.
 */
void ClassifierList::chooseBestQRAction(Action *act)
{
  double max=-100000;
  Classifier *maxCl=0;

  if(list!=0){
    for(PureClassifierList *listp = list; listp!=0; listp=listp->next){
      if(max < listp->cl->getQuality() * listp->cl->getRewardPrediction() && listp->cl->doesAnticipateChange()) {
	max = listp->cl->getQuality() * listp->cl->getRewardPrediction();
	maxCl = listp->cl;
      }
    }
  }
  if(maxCl!=0)
    act->setAction(maxCl->getAction());
  else
    chooseRandomAction(act);
}

/**
 * Chooses one of the possible actions in the environment randomly
 */
void ClassifierList::chooseRandomAction(Action *act)
{
  act->setAction( (int)(frand()*(double)env->getNoActions()));
}


/**
 * Reinforcement Learning in the list. Applies RL according to current reinforcement rho
 * and backpropagated reinforcement P.
 */
void ClassifierList::applyReinforcementLearning(double rho, double P)
{
  for(PureClassifierList *listp = list; listp!=0; listp=listp->next){
    listp->cl->updateReward(rho+GAMMA*P);
    listp->cl->updateImmediateReward(rho);
  }
}


/**
 * Executes a one-step mental acting algorithm as specified in the SAB'2000 paper
 */
void ClassifierList::doOneStepMentalActing(int steps)
{
  ClassifierList *relList = new ClassifierList(this, THETA_R);

  PureClassifierList *listp, *listpl;
  //remove all classifier that anticipate no change
  for(listp=relList->list, listpl=relList->list; listp!=0;){
    if(!listp->cl->doesAnticipateChange()){
      relList->size--;
      if(listp==relList->list){
	listp=listp->next;
	listpl->next=0;
	delete listpl;
	relList->list=listp;
	listpl=listp;
      }else{
	listpl->next=listp->next;
	listp->next=0;
	delete listp;
	listp=listpl->next;
      }
    }else{
      listpl=listp;
      listp=listp->next;
    }
  }

  if(relList->size>0){
    int i,j;
    for(i=0; i<steps; i++){
      int chosen = (int)(frand() * relList->size);
      for(j=0, listp=relList->list; j!=chosen; ++j, listp=listp->next);//directly sets to chosen classifier
      ClassifierList *linkSet = new ClassifierList(this, listp->cl);
      if(linkSet->size>0){
	double P = linkSet->getMaximumQR();
	listp->cl->updateReward( listp->cl->getImmediateRewardPrediction()+GAMMA*P);
      }
      delete linkSet;
    }
  }
  delete relList;
}



/**
 * Sets the GA time stamps to the current time to control the GA application frequency.
 */ 
void ClassifierList::setGATimeStamps(int time)
{
  for(PureClassifierList *listp = list; listp !=0; listp=listp->next){
    listp->cl->setGATimeStamp(time);
  }
}


/**
 * Sets the ALP time stamp to monitor the frequency of application and the last application.
 * The called method setALPTimeStamp also sets the application average parameter.
 */
void ClassifierList::setALPTimeStamps(int time)
{
  for(PureClassifierList *listp = list; listp !=0; listp=listp->next){
    listp->cl->setALPTimeStamp(time);
  }
}

/**
 * The Anticipatory Learning Process. Handles all updates by the ALP, 
 * insertion of new classifiers in pop and possibly matchSet, and
 * deletion of inadequate classifiers in pop and possibly matchSet.
 */
void ClassifierList::applyALP(Perception *p0, Action *act, Perception *p1, int time, ClassifierList *pop, ClassifierList *matchSet)
{
  ClassifierList *newList=0;
  PureClassifierList *listp, *listpl;

  setALPTimeStamps(time);

  int expectedCase=0;
  Classifier *newCl=0;
  for(listp=list, listpl=0; listp!=0; ){
    listp->cl->increaseExperience();

    if(listp->cl->doesAnticipateCorrect(p0, p1)) {
      newCl = listp->cl->expectedCase(p0, time);
      expectedCase=1;
    }else{
      newCl = listp->cl->unexpectedCase(p0, p1, time);
      if(listp->cl->getQuality() < THETA_I){
	assert(pop->deleteClassifierPointer(listp->cl)==1);
	if(matchSet!=0)
	  matchSet->deleteClassifierPointer(listp->cl);
	if(listpl==0){
	  list=list->next;
	  listp->next=0;
	  delete listp->cl;
	  delete listp;
	}else{
	  listpl->next = listp->next;
	  listp->next=0;
	  delete listp->cl;
	  delete listp;
	}
	listp=listpl; //Set listp to the last observed classifier list pointer
      }
    }

    if(newCl != 0)//New Classifier was created
      insertALPOffspringToNewList(newCl, &newList); //directly inserts the new classifier wherever appropriate
    
    listpl = listp;
    if(listp==0)
      listp=list;
    else
      listp=listp->next;
  }
  //Look for possible effect part enhancements if activated.
  if(DO_PEES)
    applyEnhancedEffectPartCheck(&newList, p0, time);

  //If no classifier anticipated correctly, a new one is generated.
  if(expectedCase==0){
    newCl = new Classifier(p0, act, p1, time);
    insertALPOffspringToNewList(newCl, &newList); //directly inserts the new classifier wherever appropriate
  }
  
  addClassifierList(newList);
  pop->addClassifierList(newList);
  if(matchSet != 0)
    matchSet->addMatchingClassifiers(newList, p1); //This must be 'p1' because this is the next matchSet!
  delete newList;
}


/**
 * Check if effect parts need to be enhanced to cope with stochastic environments.
 * Actual enhancements are added to newList.
 */
void ClassifierList::applyEnhancedEffectPartCheck(ClassifierList **newList, Perception *p0, int time)
{
  ClassifierList *candList = new ClassifierList();
  PureClassifierList *listp=0;
  for(listp=list; listp!=0; listp=listp->next){
    if(listp->cl->isEnhanceable())
      candList->addClassifier(listp->cl);
  }
  if(candList->size < 2){
    delete candList;
    return;
  }

  for(listp=candList->list; listp!=0; listp=listp->next){
    ClassifierList *candList2 = new ClassifierList();
    int candNum=0;
    for(PureClassifierList *listp2=candList->list; listp2!=0; listp2=listp2->next){
      if(listp != listp2 && !listp->cl->getPMark()->isEnhanced() && listp->cl->getPMark()->isEqual(listp2->cl->getPMark(),p0)) {
	candList2->addClassifier(listp2->cl);
	candNum++;
      }
    }
    if(candNum != 0){
      PureClassifierList *merger=candList2->list;
      for(candNum = (int)((double)candNum * frand()); candNum != 0; candNum--)
	merger = merger->next;
      Classifier *newCl = listp->cl->mergeClassifiers(merger->cl, p0, time);
      if(newCl != 0){
	listp->cl->reverseIncreaseQuality();
	insertALPOffspringToNewList(newCl, newList); //Directly inserts ALP offspring wherever appropriate
      }
    }
    delete candList2;
  }
  delete candList;
}


/**
 * Checks the average last GA application to determine if a GA should be applied.
 * If no classifier is in the current set, no GA is applied!
 */
int ClassifierList::doApplyGA(int time)
{
  int t=0, num=0; 

  for(PureClassifierList *listp=list; listp!=0; listp=listp->next){
    t +=  listp->cl->getGATimeStamp() * listp->cl->getNumerosity();
    num += listp->cl->getNumerosity();
  }

  if(num==0)
    return 0;

  if( time - t/num > THETA_GA)
    return 1;

  return 0;
}

/**
 * Select two parents for the GA with roulette-wheel selection.
 */
void ClassifierList::selectParents(Classifier **parent1, Classifier **parent2)
{
  double qSum;
  PureClassifierList *listp;
  for(listp=list, qSum=0; listp!=0; listp=listp->next)
    qSum += listp->cl->getQuality() * listp->cl->getQuality() * listp->cl->getQuality() * listp->cl->getNumerosity();

  double qSel1 = frand() * qSum;
  double qSel2 = frand() * qSum;
  
  if(qSel1 > qSel2){
    double help = qSel1;
    qSel1 = qSel2;
    qSel2 = help;
  }
  
  double qCounter;
  for(listp=list, qCounter=0; listp!=0; listp=listp->next){
    qCounter += listp->cl->getQuality() * listp->cl->getQuality() * listp->cl->getQuality() * listp->cl->getNumerosity();
    if(qCounter > qSel1){
      if(qSel2 != -1){
	*parent1 = listp->cl;
	if(qCounter > qSel2){
	  *parent2 = listp->cl;
	  break;
	}
	qSel1 = qSel2;
	qSel2 = -1;
      }else{
	*parent2 = listp->cl;
	break;
      }
    }
  }
}

/**
 * Looks for subsuming / similar classifiers. If no appropriate classifier was found, 
 * the offspring is added to newList. Returns if an appropriate, old classifier was found.
 * This method is called from the ALP application.
 */
int ClassifierList::insertALPOffspringToNewList(Classifier *child, ClassifierList **newList)
{
  Classifier *oldCl=0;
  if( (!DO_SUBSUMPTION || (oldCl = getSubsumer(child)) == 0 ) && //check if subsumer exists (cannot be a new one)
      (*newList==0 || (oldCl = (*newList)->getSimilar(child))==0 ) &&//check if it was already created
      ((oldCl = getSimilar(child))==0)) {//check if it already existed
    if(*newList==0){
      (*newList) = new ClassifierList();
      (*newList)->env = env;
    }
    (*newList)->addClassifier(child);
    return 0;
  }else{ //old, similar/subsuming classifier was found
    delete child;
    oldCl->increaseQuality();
  }
  return 1;
}

/**
 * Looks for similar or subsuming classifier. 
 * If found, the child is deleted and if the old classifier is not marked, 
 * the numerosity of the old classifier is increased. Otherwise, nothing is changes and the 
 * method returns 0.
 * Returns if an old classifier was found. 
 */
int ClassifierList::checkGASimilarities(Classifier *child)
{
  Classifier *oldCl=0;

  if(DO_SUBSUMPTION){
   if((oldCl = getSubsumer(child)) != 0){
      oldCl->increaseNumerosity();
      delete child;
      return 1;
    }
  }

  if( (oldCl = getSimilar(child)) != 0){
    if(!oldCl->isMarked()) {
      oldCl->increaseNumerosity();
      delete child;
      return 1;
    }else{
      delete child;
      return 1;
    }
  }
  return 0;
}

/**
 * The Genetic Generalization mechanism. Handles creation, modification, and insertion of 
 * GA offspring. Also applies the genetic deletion process to the set. Keeps pop and matchSet 
 * updated.
 */
void ClassifierList::applyGA(int time, ClassifierList *pop, ClassifierList *matchSet, Perception *p1)
{
  if(doApplyGA(time)){
    setGATimeStamps(time);

    //select classifiers
    Classifier *parent1=0, *parent2=0;
    selectParents(&parent1, &parent2);

    Classifier *child1, *child2;
    child1 = new Classifier(parent1, time);
    child2 = new Classifier(parent2, time);

    child1->mutate();
    child2->mutate();
    
    if(frand() < CHI){
      if( child1->getEffect()->isEqual(child2->getEffect())) 
	child1->crossover(child2);
    }
    child1->halfQuality();
    child2->halfQuality();

    int childNo = 2;
    if( child1->getCondition()->getSpecificity()==0 ) {
      /* do not insert completely general children*/
      delete child1;
      child1 = 0;
      childNo--;
    }
    if( child2->getCondition()->getSpecificity()==0) {
      /* do not insert completely general children*/
      delete child2;
      child2 = 0;
      childNo--;
    }

    if(child1 != 0 && child2 != 0){
      if(child1->isSimilar(child2)){
	//From two identical children only one is inserted.
	delete child2;
	child2 =0;
	childNo--;
      }
    }

    //Delete classifiers in this set considering that childNo children will still be added
    deleteGAClassifiers(pop, matchSet, childNo);

    //check for subsumer / similar classifier
    if( child1!=0 && checkGASimilarities(child1) )
      child1=0;
    if( child2!=0 && checkGASimilarities(child2) )
      child2=0;

    //finally, add the children if they were no already subsumed or represented by old classifiers
    if(child1){
      addClassifier(child1);
      pop->addClassifier(child1);
      if(matchSet!=0)
	matchSet->addMatchingClassifier(child1, p1);
    }
    if(child2){
      addClassifier(child2);
      pop->addClassifier(child2);
      if(matchSet!=0)
	matchSet->addMatchingClassifier(child2, p1);
    }
  }
}

/**
 * Deletes classifiers in the set to keep the size THETA_AS. Also condisers that still
 * childNo classifiers are added by the GA. 
 */ 
void ClassifierList::deleteGAClassifiers(ClassifierList *pop, ClassifierList *set, int childNo)
{
  int delNo = getNumSize() + childNo - THETA_AS;
  if(delNo<=0)//There is still room for more classifiers
    return;
  
  //delNo Classifiers will be deleted
  for(int i=0; i<delNo; i++){
    Classifier *delCl=0;
    PureClassifierList *listp;
    int j;
    for( j=0, listp=list; listp!=0; j++){
      //We do consider the micro-classifiers here! (see if clause later)
      if(frand() < 1./3.){
	if(delCl==0){ 
	  delCl=listp->cl;
	}else{
	  if( listp->cl->getQuality() - delCl->getQuality() < -0.1 ||
	      (listp->cl->getQuality() - delCl->getQuality() < 0.1 &&
	       ( (listp->cl->isMarked() && !delCl->isMarked()) ||
		 ( !(!listp->cl->isMarked() && delCl->isMarked()) &&
		   listp->cl->getApplicationAverage() > delCl->getApplicationAverage() 
		   //&& 
		   //frand()<0.5
		   ))))
	    delCl = listp->cl;
	}
      }
      if(j+1 == listp->cl->getNumerosity()){
	j=-1;
	listp = listp->next;
      }
    }
    if(delCl==0){
      i--;
    }else{
      if(delCl->decreaseNumerosity() == 0) {
	assert(deleteClassifierPointer(delCl)!=0);
	assert(pop->deleteClassifierPointer(delCl)!=0);
	if(set!=0)
	  set->deleteClassifierPointer(delCl);
	delete delCl;
      }
    }
  }
}

/**
 * Deletes Classifier Pointer from the list.
 * Returns if the pointer was found and deleted.
 */
int ClassifierList::deleteClassifierPointer(Classifier *clp)
{
  PureClassifierList *listp, *listpl;
  for(listp=list, listpl=0; listp!=0; listp=listp->next){
    if(listp->cl == clp){
      if(listpl==0){
	list = listp->next;
      }else{
	listpl->next = listp->next;
      }
      listp->next=0;
      delete listp;
      size--;
      return 1;
    }
    listpl=listp;
  }
  return 0;
}

/**
 * Delete all classifiers in the population. 
 * Note that the population needs to be destroyed immediately afterwards to prevent errors.
 */
void ClassifierList::deleteClassifiers()
{
  for(PureClassifierList *listp=list; listp!=0; listp=listp->next)
    delete listp->cl;
}

/**
 * Adds Classifier cl to the list (in front).
 */
void ClassifierList::addClassifier(Classifier *cl)
{
  PureClassifierList* listp = new PureClassifierList(cl);
  listp->next = list;
  list = listp;
  size++;
}


/**
 * Returns similar classifier if is contained in the list, 0 otherwise.
 */
Classifier *ClassifierList::getSimilar(Classifier *cl)
{
  for(PureClassifierList *listp=list; listp!=0; listp=listp->next){
    if(listp->cl->isSimilar(cl))
      return listp->cl;
  }
  return 0;
}

/**
 * Returns subsuming classifier if is contained in the list, 0 otherwise.
 */
Classifier *ClassifierList::getSubsumer(Classifier *cl)
{
  Classifier *subsumer=0;
  ClassifierList *subsList = 0;
  
  PureClassifierList *listP=0;
  for(listP=list; listP!=0; listP=listP->next) {
    if(listP->cl->doesSubsume(cl)) {
      if(subsumer == 0){
	subsumer = listP->cl;
	subsList = new ClassifierList();
	subsList->addClassifier(subsumer);
      }else if(listP->cl->isMoreGeneral(subsumer)){
	//another more general subsumer was found
	delete subsList;
	subsumer = listP->cl;
	subsList = new ClassifierList();
	subsList->addClassifier(subsumer);
      }else if(!subsumer->isMoreGeneral(listP->cl)){
	//another subsumer with equal generality was found
	subsList->addClassifier(listP->cl);
      }
    }
  }
  if(subsList != 0){
    // choose randomly one subsumer among the most-general subsumers
    int select = (int)(frand() * (double)subsList->getNumSize());
    listP=subsList->list;
    for(int i=listP->cl->getNumerosity(); i<=select; i += listP->cl->getNumerosity() ) {
      listP=listP->next;
    }
    subsumer = listP->cl;
    delete subsList;
    return subsumer;
  }
  return 0;
}


/**
 * Returns the number of macroclassifiers in the population.
 */
int ClassifierList::getSize() 
{
  return size;
}

/**
 * Returns the number of micro-classifiers in the population.
 */
int ClassifierList::getNumSize()
{
  int numSum=0;
  for(PureClassifierList *listp=list; listp!=0; listp=listp->next){
    numSum += listp->cl->getNumerosity();
  }
  return numSum;
}


/**
 * Returns the returns the average specifictiy of the conditions in the population.
 */
double ClassifierList::getSpecificity()
{
  int numSum=0;
  double spec=0;
  for(PureClassifierList *listp=list; listp!=0; listp=listp->next){
    spec += listp->cl->getSpecificity()*listp->cl->getNumerosity();
    numSum += listp->cl->getNumerosity();
  }
  return spec/numSum;
}



/**
 * Searches a path from start to goal using a bidirectional method in the environmental model 
 * (i.e. the list of reliable classifiers). 
 */
Action** ClassifierList::searchGoalSequence(Perception *start, Perception *goal)
{
  //cout<<"Searching for sequence from "<<start<<" to "<<goal<<endl;

  ClassifierList *relList = new ClassifierList(this, THETA_R);
  ClassifierList **arrayListF = new ClassifierList*[10000];
  ClassifierList **arrayListB = new ClassifierList*[10000];
  //the classifier list array is one behind the perception array
  Perception **arrayPF = new Perception*[10001];
  Perception **arrayPB = new Perception*[10001];
  int maxDepth=6; //(=12)
  int fSize=1, bSize=1;
  int fPoint=0, bPoint=0;
  arrayPF[0]= new Perception(start);
  arrayPB[0]= new Perception(goal);

  Action **actSeq=0;

  for(int depth=0; depth<maxDepth; depth++){
    //Forward Step
    int fs=fSize; //records the new place of fSize;
    actSeq=searchOneForwardStep(relList, &fs, fSize, fPoint, arrayPF, bSize, arrayPB, arrayListF, arrayListB);
    fPoint=fSize;
    fSize=fs;

    if(actSeq!=0){//Some sequence was returned -> free stuff and return sequence
      freeBidisearchStuff(arrayListF, arrayPF, fSize, arrayListB, arrayPB, bSize, relList);
      return actSeq;
    }

    //Backwards Step
    int bs=bSize; //records the new place of fSize;
    actSeq=searchOneBackwardStep(relList, &bs, bSize, bPoint, arrayPB, fSize, arrayPF, arrayListB, arrayListF);
    bPoint=bSize;
    bSize=bs;

    if(actSeq!=0){//Some sequence was returned -> free stuff and return sequence
      freeBidisearchStuff(arrayListF, arrayPF, fSize, arrayListB, arrayPB, bSize, relList);
      return actSeq;
    }
  }
  //Depth limit is reached -> free stuff and return an action array with a 0 entry
  freeBidisearchStuff(arrayListF, arrayPF, fSize, arrayListB, arrayPB, bSize, relList);
  actSeq=new Action*[1];
  actSeq[0]=0;
  return actSeq;
}

/**
 * Frees the allocated memory for the bidirectional search method.
 */ 
void ClassifierList::freeBidisearchStuff(ClassifierList **al1, Perception **ap1, int size1, ClassifierList **al2, Perception **ap2, int size2, ClassifierList *cllist)
{
  int i;
  for(i=0; i<size1-1; i++){
    delete al1[i];
    delete ap1[i];
  }
  delete ap1[i];
  delete[] al1;
  delete[] ap1;

  for(i=0; i<size2-1; i++){
    delete al2[i];
    delete ap2[i];
  }
  delete ap2[i];
  delete[] al2;
  delete[] ap2;

  delete cllist;
}

/**
 * Returns the position in the perception 'ap' where 'state' is stored or 0 if state is not found.
 */ 
int ClassifierList::doesContainState(Perception **ap, int asize, Perception *state)
{
  for(int i=0; i<asize; i++){
    if(ap[i]->isEqual(state))
      return i;
  }
  return -1;
}

/** 
 * Searches one step forward in the relList classifier list. 
 * Returns 0 if nothing was found so far, a sequence with a 0 element if the search failed completely 
 * (which is the case if the allowed array size of 10000 is reached), and the sequence if one was found.
 */
Action **ClassifierList::searchOneForwardStep(ClassifierList *relList, int *fs, int fSize, int fPoint, Perception **arrayPF, 
					      int bSize, Perception **arrayPB, ClassifierList **arrayListF, ClassifierList **arrayListB)
{
  for(int i=fPoint; i<fSize; i++){
    ClassifierList * matchFW = new ClassifierList(relList, arrayPF[i]);
    if(matchFW->size > 0){
      //matching classifiers found
      for(PureClassifierList *listp=matchFW->list; listp!=0; listp=listp->next){
	Perception *ant = listp->cl->getBestAnticipation(arrayPF[i]);
	if( doesContainState(arrayPF,*fs,ant)==-1 ){//state not detected forward -> search in backwards
	  int bwseq=-1;
	  if((bwseq=doesContainState(arrayPB, bSize, ant))==-1){//state neither detected backwards
	    //add classifier List and new state
	    arrayPF[(*fs)]=ant;
	    if(i>0)
	      arrayListF[(*fs)-1] = new ClassifierList(arrayListF[i-1]);
	    else
	      arrayListF[(*fs)-1]= new ClassifierList();
	    arrayListF[(*fs)-1]->addClassifier(listp->cl);
	    (*fs)++;
	    if(*fs > 10001){
	      cout<<"Arrays are full!"<<endl;
	      delete ant;
	      delete matchFW;
	      Action **actSeq=new Action*[1];
	      actSeq[0]=0;
	      return actSeq;
	    }
	  }else{//sequence found!
	    int seqSize=0;
	    if(i>0)
	      seqSize += arrayListF[i-1]->size;
	    if(bwseq>0)
	      seqSize += arrayListB[bwseq-1]->size;
	    seqSize+=2;
	    Action **actSeq= new Action*[seqSize];
	    actSeq[seqSize-1]=0;
	    //construct sequence here! and free stuff
	    int j=0;
	    if(i>0){
	      PureClassifierList *pclf=arrayListF[i-1]->list;
	      for(j=0; j<arrayListF[i-1]->size; j++){
		actSeq[(arrayListF[i-1]->size) - j - 1]=new Action(pclf->cl->getAction());
		pclf=pclf->next;
	      }
	    }
	    actSeq[j]=new Action(listp->cl->getAction());
	    j++;
	    if(bwseq>0){
	      PureClassifierList *pclb=arrayListB[bwseq-1]->list;
	      for(int k=0; k < arrayListB[bwseq-1]->size; k++){
		actSeq[k+j]=new Action(pclb->cl->getAction());
		pclb=pclb->next;
	      }
	    }
	    delete ant;
	    delete matchFW;
	    return actSeq;
	  }
	}else{//state was already detected -> can be reached by a shorter sequence of steps
	  delete ant;
	}
      }
    }
    delete matchFW;
  }
  return 0;
}

/** 
 * Searches one step backward in the relList classifier list. 
 * Returns 0 if nothing was found so far, a sequence with a 0 element if the search failed completely
 * (which is the case if the allowed array size of 10000 is reached), and the sequence if one was found.
 */
Action **ClassifierList::searchOneBackwardStep(ClassifierList *relList, int *bs, int bSize, int bPoint, Perception **arrayPB, 
					      int fSize, Perception **arrayPF, ClassifierList **arrayListB, ClassifierList **arrayListF)
{
  for(int i=bPoint; i<bSize; i++){
    ClassifierList * matchBW = new ClassifierList(arrayPB[i], relList);
    if(matchBW->size > 0){
      //matching classifiers found
      for(PureClassifierList *listp=matchBW->list; listp!=0; listp=listp->next){
	Perception *ant = listp->cl->getBackwardsAnticipation(arrayPB[i]);
	if(ant!=0){//backwards anticipation was formable
	  if( doesContainState(arrayPB,*bs,ant)==-1 ){//state not detected backward -> search in forward list
	    int fwseq=-1;
	    if((fwseq=doesContainState(arrayPF, fSize, ant))==-1){//state neither detected forward
	      //add classifier List and new state
	      arrayPB[(*bs)]=ant;
	      if(i>0)
		arrayListB[(*bs)-1] = new ClassifierList(arrayListB[i-1]);
	      else
		arrayListB[(*bs)-1]= new ClassifierList();
	      arrayListB[(*bs)-1]->addClassifier(listp->cl);
	      (*bs)++;
	      if(*bs > 10001){
		cout<<"Arrays are full!"<<endl;
		delete ant;
		delete matchBW;
		Action **actSeq=new Action*[1];
		actSeq[0]=0;
		return actSeq;
	      }
	    }else{//sequence found!
	      int seqSize=0;
	      if(i>0)
		seqSize += arrayListB[i-1]->size;
	      if(fwseq>0)
		seqSize += arrayListF[fwseq-1]->size;
	      seqSize+=2;
	      Action **actSeq= new Action*[seqSize];
	      actSeq[seqSize-1]=0;
	      int j=0;
	      if(fwseq>0){
		PureClassifierList *pclf=arrayListF[fwseq-1]->list;
		for(j=0; j<arrayListF[fwseq-1]->size; j++){
		  actSeq[(arrayListF[fwseq-1]->size) -j -1]=new Action(pclf->cl->getAction());
		  pclf=pclf->next;
		}
	      }
	      actSeq[j]=new Action(listp->cl->getAction());
	      j++;
	      if(i>0){
		PureClassifierList *pclb=arrayListB[i-1]->list;
		for(int k=0; k<arrayListB[i-1]->size; k++){
		  actSeq[k+j]=new Action(pclb->cl->getAction());
		  pclb=pclb->next;
		}
	      }
	      delete ant;
	      delete matchBW;
	      return actSeq;
	    }
	  }else{//state was already detected from backwards -> a shorter sequence is available
	    delete ant;
	  }
	}
      }
    }
    delete matchBW;
  }
  return 0;
}




/**
 * Prints the ClassifierList cll.
 */ 
ostream& operator<<(ostream& out, ClassifierList *cll)
{  
  if(cll->list!=0)
    out<<cll->list;
  return out;
}

/**
 * Prints the PureClassifierList pcll.
 */
ostream& operator<<(ostream& out, PureClassifierList *pcll)
{  
  for(PureClassifierList *listp=pcll; listp!=0; listp=listp->next){
    out<<listp->cl<<endl;
  }
  return out;
}
