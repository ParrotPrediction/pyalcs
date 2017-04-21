/*
/       ACS2 in C++
/	------------------------------------
/       choice without/with GA, subsumption, PEEs
/
/     (c) by Martin V. Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 02-23-2001
/
/     main program
*/

#include<iostream>
#include<fstream>

#include <unistd.h>

#include"Perception.h"
#include"Action.h"
#include"ACSConstants.h"
#include"Classifier.h"
#include"ClassifierList.h"
#include"MazeEnvironment.h"

using namespace std;

int Perception::length = 0;
Environment* Action::env = 0;

/**
 * Keeps the maximum knowedge percentage reached so far.
 */
int knowledge;

void startExperiments(Environment *env);
void startOneExperiment(Environment *env, ofstream *out);
int startOneTrialExplore(ClassifierList *population, Environment *env, int time, ofstream *out);
int startOneTrialExploit(ClassifierList *population, Environment *env);
int startActionPlanning(ClassifierList *population, Environment *env, int time, ofstream *out, Perception *situation, Perception *previousSituation, ClassifierList **actionSet, Action *act, double *rho0);
void startCRRatExperiment(Environment *env, ofstream *out);
void printTestSortedClassifierList(ClassifierList *list, Environment *env, ofstream *out);
void testModel(ClassifierList *pop, ofstream *out, Environment *env, int time);
void testList(ClassifierList *pop, ofstream *out, Environment *env, int time);
void writeRewardPerformance(ClassifierList *pop, int *steps, int time, int trial, ofstream *out);
void randomize(void);

/**
 * main requires the input of one default parameter. 
 * This is a maze file in a MazeEnvironment run and nothing otherwise.
 */
int main(int args, char *argv[])
{
  /* set the priority */
  setpriority(PRIO_PROCESS, getpid(), 0);

  randomize();

  ENVIRONMENT_CLASS *env;
  if(args==2){
    env=new ENVIRONMENT_CLASS(argv[1]);
  }else{
    cout<<"usage: acs++.out (MazeFile)Name"<<endl;
    exit(0);
  }

  Perception::length = env->getPerceptionLength();
  Action::env = env;

  startExperiments(env);

  delete env;
}

/**
 * Controls the execution of the specified number of experiments.
 */
void startExperiments(Environment *env)
{
  ofstream *out = new ofstream( RESULT_FILE, ios::out);

  *out<<"# beta: "<<BETA<<" gamma: "<<GAMMA<<" theta_i: "<<THETA_I<<" theta_e: "<<THETA_R<<" r_ini: "<<R_INI<<" q_ini:"<<Q_INI<<" avt_ini: "<<AVT_INI<<" q_alp_min: "<<Q_ALP_MIN<<" q_ga_decrease: "<<Q_GA_DECREASE<<endl;
  *out<<"# umax: "<<U_MAX<<" doPees: "<<DO_PEES<<" epsilon: "<<EPSILON<<" prob_exploration_bias: "<<PROB_EXPLORATION_BIAS<<" exploration bias method: "<<EXPLORATION_BIAS_METHOD<<" do_action_planning: "<<DO_ACTION_PLANNING<<" action_planning_frequency: "<<ACTION_PLANNING_FREQUENCY<<endl;
  *out<<"# do_ga: "<<DO_GA<<" theta_ga: "<<THETA_GA<<" mu: "<<MU<<" X.type: "<<X_TYPE<<" chi: "<<CHI<<" theta_as: "<<THETA_AS<<" theta_exp: "<<THETA_EXP<<" do_subsumption: "<<DO_SUBSUMPTION<<endl;
  *out<<"# max_steps: "<<MAX_STEPS<<" max_trial_steps: "<<MAX_TRIAL_STEPS<<" anz_experiments: "<<ANZ_EXPERIMENTS<<" reward_test: "<<REWARD_TEST<<" model_test_iteration: "<<MODEL_TEST_ITERATION<<" reward_test_iteration: "<<REWARD_TEST_ITERATION<<endl;

  char *id=env->getID();
  
  for(int i=0; i<ANZ_EXPERIMENTS; i++){
    *out<<"Next Experiment"<<endl;
    cout<<"Experiment Nr: "<<(i+1)<<endl;
    if(strcmp(id,"CRRat")==0){
      startCRRatExperiment(env,out);
    }else{
      startOneExperiment(env, out);
    }
  }
  
  delete[] id;
  delete out;
}

/**
 * Controls the execution of one experiment.
 */
void startOneExperiment(Environment *env, ofstream *out)
{
  int time=0, trial, exploitSteps[REWARD_TEST_ITERATION];
  
  for(trial=0; trial<REWARD_TEST_ITERATION; trial++){
    char *id=env->getID();
    if(strcmp(id,"MP")==0)
      exploitSteps[trial]=0;
    else
      exploitSteps[trial] = REWARD_TEST_ITERATION;
    delete[] id;
  }

  ClassifierList *population = new ClassifierList(env);
  cout<<population;
  
  knowledge=0;
  trial=0;

  while((REWARD_TEST || time<=MAX_STEPS) && (!REWARD_TEST || trial<=MAX_STEPS)){
    time += startOneTrialExplore( population, env, time, out);
    if(REWARD_TEST){
      exploitSteps[trial%REWARD_TEST_ITERATION] = startOneTrialExploit(population, env);
      if(trial%REWARD_TEST_ITERATION == 0){
	writeRewardPerformance(population, exploitSteps, time, trial, out);
      }
    }
    trial++;
  }
  //printTestSortedClassifierList(population, env, out);
  //*out<<population<<endl;
  
  population->deleteClassifiers();
  delete population;
}

/**
 * Controls the execution of one exploration (learning) trial. The environment specifies when one trial ends.
 */
int startOneTrialExplore(ClassifierList *population, Environment *env, int time, ofstream *out)
{
  ClassifierList *matchSet, *actionSet=0;
  int steps;
  double rho0=0;

  Perception *situation = new Perception();
  Perception *previousSituation = new Perception();

  env->reset();
  env->getSituation(situation);

  Action *act=new Action();

  for(steps=0; !env->isReset() && (REWARD_TEST || time+steps<=MAX_STEPS) && (!REWARD_TEST || steps<MAX_TRIAL_STEPS); steps++) {

    if(!REWARD_TEST && (time+steps)% MODEL_TEST_ITERATION == 0){
      if(MODEL_TEST_TYPE==0)
	testModel(population, out, env, time+steps);
      else
	testList(population, out, env, time+steps);
    }

    if(DO_MENTAL_ACTING_STEPS>0)
      population->doOneStepMentalActing(DO_MENTAL_ACTING_STEPS);

    if( DO_ACTION_PLANNING && (time+steps)%ACTION_PLANNING_FREQUENCY==0){
      // Action planning for increased model learning.
      char *id = env->getID();
      if(strcmp(id,"HandEye")==0)
	steps += startActionPlanning(population, env, time+steps, out, situation, previousSituation, &actionSet, act, &rho0);
      delete[] id;
    }

    matchSet = new ClassifierList(population, situation);

    if(steps>0){
      //Learning in the last action set.
      actionSet->applyALP(previousSituation, act, situation, time+steps, population, matchSet);
      
      actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR() );
      if(DO_GA){
	actionSet->applyGA(time+steps, population, matchSet, situation);
      }
      delete actionSet;
    }

    matchSet->chooseAction(act, population, situation);
    actionSet = new ClassifierList(matchSet, act);
    delete matchSet;
    
    rho0 = env->executeAction(act);
    
    previousSituation->setPerception(situation);
    env->getSituation(situation);

    if(env->isReset()){
      //Learning in the current action set if end of trial.
      actionSet->applyALP(previousSituation, act, situation, time+steps, population, 0);
      actionSet->applyReinforcementLearning(rho0, 0);
      if(DO_GA){
	actionSet->applyGA(time+steps, population, 0, situation);
      }
    }
  }
  delete actionSet;
  delete situation;
  delete previousSituation;
  delete act;

  return steps;
}

/**
 * Executes on explotation trial. 
 * Here always the apparent best action (i.e. max(q*r)) is executed.
 */
int startOneTrialExploit(ClassifierList *population, Environment *env)
{
  ClassifierList *matchSet, *actionSet;
  int steps;
  double rho0=0;

  Perception *situation = new Perception();

  env->reset();
  env->getSituation(situation);

  Action *act=new Action();

  for(steps=0; !env->isReset() && steps<MAX_TRIAL_STEPS; steps++) {

    matchSet = new ClassifierList(population, situation);

    if(steps>0){
      //Reinforcement learning also during exploitation
      actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR() );
      delete actionSet;
    }

    matchSet->chooseBestQRAction(act);
    actionSet = new ClassifierList(matchSet, act);
    delete matchSet;
    
    rho0 = env->executeAction(act);
    
    env->getSituation(situation);

    if(env->isReset()) {
      actionSet->applyReinforcementLearning(rho0, 0);
    }
  }
  delete actionSet;
  delete situation;

  char *id=env->getID();
  if(strcmp(id,"MP")==0){
    if(rho0<1)
      return 0;
  }
  delete[] id;

  return steps;
}


/**
 * Executes action planning for model learning speed up. 
 * Method requests goals from a 'goal generator' provided by the environment. 
 * If goal is provided, ACS2 searches for a goal sequence in the current model (only the reliable classifiers). 
 * This is done as long as goals are provided and ACS2 finds a sequence and successfully reaches the goal.
 * Performance monitoring also works in this method.
 */
int startActionPlanning(ClassifierList *population, Environment *env, int time, ofstream *out, Perception *situation, Perception *previousSituation, ClassifierList **actionSet, Action *act, double *rho0)
{
  /* Recheck this function -> is step set correct and we need to do model testing in here! */
  Perception *goalSituation = new Perception();

  int steps;
  ClassifierList *matchSet=0;
  
  for(steps=0;  !env->isReset() && (REWARD_TEST || time+steps<=MAX_STEPS) && (!REWARD_TEST || steps<MAX_TRIAL_STEPS); ) {  

    if(!env->getGoalState(goalSituation)){
      break;
    }else{
      Action **actSequence = population->searchGoalSequence(situation, goalSituation);
      int i;
      //cout<<situation<<"->";
      //for(act=actSequence[0], i=0; act!=0; ++i, act=actSequence[i])
      // cout<<act<<"->";
      //cout<<goalSituation<<endl;

      //Execute the found sequence and learn during executing
      for(i=0; actSequence[i]!=0; i++, steps++){
	
	matchSet = new ClassifierList(population, situation);
	
	if((*actionSet) != 0){
	  (*actionSet)->applyALP(previousSituation, act, situation, time+steps, population, matchSet);
	  (*actionSet)->applyReinforcementLearning(*rho0, matchSet->getMaximumQR() );
	if(DO_GA)
	  (*actionSet)->applyGA(time+steps, population, matchSet, situation);
	delete (*actionSet);
	}
	act->setAction(actSequence[i]);
	(*actionSet) = new ClassifierList(matchSet, act);
	delete matchSet;

	(*rho0) = env->executeAction(act);
	previousSituation->setPerception(situation);
	env->getSituation(situation);
	
	if(!REWARD_TEST && (time+steps+1)% MODEL_TEST_ITERATION == 0){
	  if(MODEL_TEST_TYPE==0)
	    testModel(population, out, env, time+steps+1);
	  else
	    testList(population, out, env, time+steps+1);
	}
	
	if(!((*actionSet)->existClassifier(previousSituation, act, situation, THETA_R)))
	  break;//no reliable classifier was able to anticipate such a change
      }
      int doBreak=0;
      if(i==0 || act!=0)
	doBreak=1;
      for(i=0; actSequence[i]!=0; i++)
	delete (actSequence[i]);
      
      delete[] actSequence;
      if(doBreak)
	break;
    }
  }
  delete goalSituation;

  return steps;
    
}


/**
 * This function is programmed to execute the colwill/rescorla rat experiments.
 * This is quite a hack and just suitable for the specific experiments:-)
 * Hereby, reset denotes when the testing phase starts.
 * during testing the returned reward denotes if the action was the better one!
 */
void startCRRatExperiment(Environment *env, ofstream *out)
{
  ClassifierList *population = new ClassifierList(env);
  cout<<population;
  
  ClassifierList *matchSet, *actionSet=0;
  int time=0;
  int steps=0;
  double rho0=0;
  Perception *situation = new Perception();
  Perception *previousSituation = new Perception();
  Action *act = new Action();

  while(env->reset()==1){
    if(DO_MENTAL_ACTING_STEPS>0)
      population->doOneStepMentalActing(DO_MENTAL_ACTING_STEPS);
    steps=0;
    env->getSituation(situation);
    while(!env->isReset()){
      matchSet = new ClassifierList(population, situation);
      
      if(steps>0){
	//Learning in the last action set.
	actionSet->applyALP(previousSituation, act, situation, time+steps, population, matchSet);
	actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR() );
	if(DO_GA){
	  actionSet->applyGA(time+steps, population, matchSet, situation);
	}
	delete actionSet;
      }
      
      matchSet->chooseAction(act, population, situation);
      actionSet = new ClassifierList(matchSet, act);
      delete matchSet;
    
      //cout<<situation<<"-"<<act<<endl;

      rho0 = env->executeAction(act);
      steps++;
      time++;

      previousSituation->setPerception(situation);
      env->getSituation(situation);

      if(env->isReset()){
	//Learning in the current action set if end of trial.
	actionSet->applyALP(previousSituation, act, situation, time+steps, population, 0);
	actionSet->applyReinforcementLearning(rho0, 0);
	if(DO_GA){
	  actionSet->applyGA(time+steps, population, 0, situation);
	}
      }
    }
    delete actionSet;
    actionSet=0;
  }

  //*out<<population<<endl;

  int testTrial=0;
  do{
    //do final phase here and record behavior
    //hereby we assume one-step problem but execute the set behavior
    if(DO_MENTAL_ACTING_STEPS>0)
      population->doOneStepMentalActing(DO_MENTAL_ACTING_STEPS);
    steps=0;
    env->getSituation(situation);
    testTrial++;
    while(!env->isReset()) {
      matchSet = new ClassifierList(population, situation);
      
      if(steps>0){
	//Learning in the last action set.
	actionSet->applyALP(previousSituation, act, situation, time+steps, population, matchSet);
	actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR() );
	if(DO_GA){
	  actionSet->applyGA(time+steps, population, matchSet, situation);
	}
	delete actionSet;
      }
      matchSet->chooseAction(act, population, situation);
      actionSet = new ClassifierList(matchSet, act);
      delete matchSet;
      
      //*out<<situation<<"-"<<act<<endl;
      
      rho0 = env->executeAction(act);
      steps++;
      time++;
      
      previousSituation->setPerception(situation);
      env->getSituation(situation);
      
      if(env->isReset()){
	//Learning in the current action set if end of trial.
	actionSet->applyALP(previousSituation, act, situation, time+steps, population, 0);
	actionSet->applyReinforcementLearning(0, 0);
	if(DO_GA){
	  actionSet->applyGA(time+steps, population, 0, situation);
	}
      }
    }
    *out<<testTrial<<" "<<rho0<<" "<<population->getSize()<<endl;
    
    delete actionSet;
    actionSet=0;
  }while(env->reset()==2);

  delete situation;
  delete previousSituation;
  delete act;

  //*out<<population<<endl;

  population->deleteClassifiers();
  delete population;
}


/**
 * Prints Classifiers that match in from the environment env requested test situations.
 * Prints to the stream out.
 */
void printTestSortedClassifierList(ClassifierList *list, Environment *env, ofstream *out)
{
  env->doTesting();
  Perception *s1 = new Perception();
  Action *a = new Action();
  Perception *s2 = new Perception();

  while(env->getNextTest(s1,a,s2)){
    ClassifierList *matchSet = new ClassifierList(list, s1);
    ClassifierList *actionSet = new ClassifierList(matchSet, a);
    *out<<s1<<"-"<<a<<"-"<<endl<<s2<<endl<<actionSet<<endl;
    delete actionSet;
    delete matchSet;
  }
  env->endTesting();
  delete s1;
  delete a;
  delete s2;
}

/**
 * Tests the current environmental model of ACS2 and writes result to stream out.
 * Testing is done by interaction with the environment that provides the test triples.
 * The reliable list is searched for a classifier that matches, specifies the action, 
 * and anticipates correctly. The parameter knowledge is global and serves for monitoring 
 * purposes. 
 */
void testModel(ClassifierList *pop, ofstream *out, Environment *env, int time)
{
  ClassifierList *relList = new ClassifierList(pop, THETA_R);
  double nrCorrect=0, nrWrong=0;
  env->doTesting();
  Perception *s1 = new Perception();
  Action *a = new Action();
  Perception *s2 = new Perception();

  while(env->getNextTest(s1,a,s2)) {
    //cout<<s1<<"-"<<a<<"-"<<endl<<s2<<endl;
    if(relList->existClassifier(s1,a,s2, THETA_R))
      nrCorrect++;
    else{
       nrWrong++;
       /*if(time>500000){
	 cout<<s1<<"-"<<a<<"-"<<endl<<s2<<endl;
	 }*/
    }
  }

  if(knowledge < nrCorrect*100/(nrCorrect+nrWrong)){
    while(knowledge < nrCorrect*100/(nrCorrect+nrWrong))
      knowledge += 2;
    cout<<"Knowlege greater than "<<knowledge<<"% at time "<<time<<endl;
  }

  cout<<time<<" "<<nrCorrect<<"-"<<nrWrong<<"="<<(nrCorrect*100/(nrCorrect+nrWrong))<<" "<<pop->getSize()<<" "<<pop->getNumSize()<<" "<<relList->getSize()<<" "<<pop->getSpecificity()<<endl;
  *out<<time<<" "<<(nrCorrect*100/(nrCorrect+nrWrong))<<" "<<pop->getSize()<<" "<<pop->getNumSize()<<" "<<relList->getSize()<<" "<<pop->getSpecificity()<<endl;

  env->endTesting();
  delete relList;
  delete s1;
  delete a;
  delete s2;
}

/**
 * Is basically the same as testModel, however, not the reliable classifiers are tested, but 
 * always the classifier with the strongest quality is selected for the test.
 */
void testList(ClassifierList *pop, ofstream *out, Environment *env, int time)
{
  double nrCorrect=0, nrWrong=0;
  env->doTesting();
  Perception *s1 = new Perception();
  Action *a = new Action();
  Perception *s2 = new Perception();

  while(env->getNextTest(s1,a,s2)) {
    ClassifierList *mset = new ClassifierList(pop, s1);
    ClassifierList *aset = new ClassifierList(mset, a);
    Classifier *testCl = aset->getHighestQualityClassifier();
    if(testCl!=0 && testCl->doesAnticipateCorrect(s1,s2)){
      nrCorrect++;
    }else{
      nrWrong++;
    }
    delete mset;
    delete aset;
  }

  if(knowledge < nrCorrect*100/(nrCorrect+nrWrong)){
    while(knowledge < nrCorrect*100/(nrCorrect+nrWrong))
      knowledge += 2;
    cout <<(nrCorrect*100/(nrCorrect+nrWrong))<<"% knowledge at time "<<time<<" with pop. size "<<pop->getSize()<<endl;
  }

  // rel list is created to determine the size of the current model.
  ClassifierList *relList = new ClassifierList(pop, THETA_R);

  cout<<time<<" "<<(nrCorrect*100/(nrCorrect+nrWrong))<<" "<<pop->getSize()<<" "<<pop->getNumSize()<<" "<<relList->getSize()<<endl;
  *out<<time<<" "<<(nrCorrect*100/(nrCorrect+nrWrong))<<" "<<pop->getSize()<<" "<<pop->getNumSize()<<" "<<relList->getSize()<<endl;

  env->endTesting();
  delete relList;
  delete s1;
  delete a;
  delete s2;
}

/**
 * Writes the RL performance of ACS2 to out. The steps needed to reach the goal during 
 * the last REWARD_TEST_ITERATION exploitation trials are reported in steps. The method averages 
 * the results and writes them to out.
 */
void writeRewardPerformance(ClassifierList *pop, int *steps, int time, int trial, ofstream *out)
{
  double performance=0;

  for(int i=0; i < REWARD_TEST_ITERATION; i++)
    performance += steps[i];

  performance /= (double)REWARD_TEST_ITERATION;
  
  // rel list is created to determine the size of the current model.
  ClassifierList *relList = new ClassifierList(pop, THETA_R);  

  cout<<trial<<" "<<performance<<" "<<pop->getSize()<<" "<<pop->getNumSize()<<" "<<relList->getSize()<<" "<<time<<endl;
  *out<<trial<<" "<<performance<<" "<<pop->getSize()<<" "<<pop->getNumSize()<<" "<<relList->getSize()<<" "<<time<<endl;  

  delete relList;
}


/**
 * Used for randomizing the random number generator.
 */
void randomize(void)
{
  int i;
  for (i=0;i<time(NULL)%1000;rand(),i++);
}

