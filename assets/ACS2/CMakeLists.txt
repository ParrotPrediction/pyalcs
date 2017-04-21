cmake_minimum_required(VERSION 3.7)
project(ACS2)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        acs2++.cc
        ACSConstants.h
        Action.cc
        Action.h
        BWEnvironment.cc
        BWEnvironment.h
        CharCharPosList.cc
        CharCharPosList.h
        CharList.cc
        CharList.h
        CharPosList.cc
        CharPosList.h
        Classifier.cc
        Classifier.h
        ClassifierList.cc
        ClassifierList.h
        Condition.cc
        Condition.h
        CRRatEnvironment.cc
        CRRatEnvironment.h
        Effect.cc
        Effect.h
        Environment.cc
        Environment.h
        GripperEnvironment.cc
        GripperEnvironment.h
        HandEyeEnvironment.cc
        HandEyeEnvironment.h
        MazeEnvironment.cc
        MazeEnvironment.h
        MPEnvironment.cc
        MPEnvironment.h
        Perception.cc
        Perception.h
        PMark.cc
        PMark.h
        ProbCharList.cc
        ProbCharList.h
        ProbCharPosList.cc
        ProbCharPosList.h)

add_executable(ACS2 ${SOURCE_FILES})