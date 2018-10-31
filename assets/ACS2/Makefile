#
#       (ACS with GA and PEEs inn C++)
#	------------------------------------
#	the Anticipatory Classifier System (ACS) with ALP in action set, GA generalization and PEEs
#
#     (c) by Martin Butz
#     University of Wuerzburg / University of Illinois at Urbana/Champaign
#     butz@illigal.ge.uiuc.edu
#     Last modified: 11-30-2000
#
#     Makefile


NAME = acs2++
LINKMATH = -lm
CC_OPTS  = -g -Wall 
#CC_opts = -O3
COMPILER = g++
EXECNAME = $(NAME).$(SYSTEM)

$(NAME):	$(NAME).o Condition.o Perception.o Action.o CharPosList.o ProbCharPosList.o ProbCharList.o Effect.o CharCharPosList.o CharList.o PMark.o Classifier.o ClassifierList.o ACSConstants.h Environment.o MazeEnvironment.o MPEnvironment.o BWEnvironment.o HandEyeEnvironment.o GripperEnvironment.o CRRatEnvironment.o
		$(COMPILER) $(CC_OPTS) -o $(NAME).out $(LINKMATH) $(NAME).o Condition.o Perception.o Action.o Effect.o CharPosList.o ProbCharPosList.o ProbCharList.o CharCharPosList.o CharList.o PMark.o Classifier.o ClassifierList.o Environment.o MazeEnvironment.o MPEnvironment.o BWEnvironment.o HandEyeEnvironment.o GripperEnvironment.o CRRatEnvironment.o

$(NAME).o:	$(NAME).cc ACSConstants.h Perception.h Condition.h Action.h Effect.h PMark.h Classifier.h ClassifierList.h Environment.h MazeEnvironment.h MPEnvironment.h BWEnvironment.h HandEyeEnvironment.h GripperEnvironment.h CRRatEnvironment.h
		$(COMPILER) -c $(CC_OPTS) $(NAME).cc

ClassifierList.o:  ClassifierList.cc ClassifierList.h Classifier.h Action.h ACSConstants.h Environment.h
		$(COMPILER) -c $(CC_OPTS) ClassifierList.cc

Perception.o:   Perception.cc Perception.h
		$(COMPILER) -c $(CC_OPTS) Perception.cc

Classifier.o:	Classifier.cc Classifier.h Condition.h Action.h Effect.h PMark.h Perception.h ACSConstants.h
		$(COMPILER) -c $(CC_OPTS) Classifier.cc

Condition.o:    Condition.cc Condition.h Perception.h CharPosList.h
		$(COMPILER) -c $(CC_OPTS) Condition.cc 

CharPosList.o:  CharPosList.cc CharPosList.h
		$(COMPILER) -c $(CC_OPTS) CharPosList.cc

Action.o:       Action.cc Action.h Environment.h
		$(COMPILER) -c $(CC_OPTS) Action.cc

Effect.o: 	Effect.cc Effect.h Perception.h ProbCharPosList.h ProbCharList.h Condition.h
		$(COMPILER) -c $(CC_OPTS) Effect.cc

ProbCharPosList.o:  ProbCharPosList.cc ProbCharPosList.h ProbCharList.h CharPosList.h
		$(COMPILER) -c $(CC_OPTS) ProbCharPosList.cc

ProbCharList.o: ProbCharList.cc ProbCharList.h
		$(COMPILER) -c $(CC_OPTS) ProbCharList.cc

PMark.o: 	PMark.cc PMark.h
		$(COMPILER) -c $(CC_OPTS) PMark.cc

CharCharPosList.o:  CharCharPosList.cc CharCharPosList.h
		$(COMPILER) -c $(CC_OPTS) CharCharPosList.cc

CharList.o: 	CharList.cc CharList.h
		$(COMPILER) -c $(CC_OPTS) CharList.cc

Environment.o:  Environment.h Environment.cc
		$(COMPILER) -c $(CC_OPTS) Environment.cc

MazeEnvironment.o:  MazeEnvironment.h MazeEnvironment.cc Environment.h Action.h Perception.h
		$(COMPILER) -c $(CC_OPTS) MazeEnvironment.cc 

MPEnvironment.o:MPEnvironment.h MPEnvironment.cc Environment.h Action.h Perception.h
		$(COMPILER) -c $(CC_OPTS) MPEnvironment.cc

BWEnvironment.o:BWEnvironment.h BWEnvironment.cc Environment.h Action.h Perception.h
		$(COMPILER) -c $(CC_OPTS) BWEnvironment.cc

HandEyeEnvironment.o:HandEyeEnvironment.h HandEyeEnvironment.cc Environment.h Action.h Perception.h
		$(COMPILER) -c $(CC_OPTS) HandEyeEnvironment.cc

GripperEnvironment.o:GripperEnvironment.h GripperEnvironment.cc Environment.h Action.h Perception.h
		$(COMPILER) -c $(CC_OPTS) GripperEnvironment.cc

CRRatEnvironment.o:CRRatEnvironment.h CRRatEnvironment.cc Environment.h Action.h Perception.h
		$(COMPILER) -c $(CC_OPTS) CRRatEnvironment.cc

clean:	
	-rm -f *.o *.out
