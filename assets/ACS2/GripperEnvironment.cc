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
/     gripper environment class
*/

#include<iostream>

#include"GripperEnvironment.h"


/**
 * Creates a blocks world environment of given number of blocks and stacks.
 */
GripperEnvironment::GripperEnvironment(char *nothing) {
    env = new char[6];

    reset();
    doReset = 0;

    cout << "Read in: " << endl << this << endl;
}


/**
 * Destructor.
 */
GripperEnvironment::~GripperEnvironment() {
    delete[] env;
}


/**
 * Sets the perception to the current perception.
 */
void GripperEnvironment::getSituation(Perception *perception) {
    perception->setPerception(env);
}


/**
 * Executes action 'act' in the blocks world. 
 */
double GripperEnvironment::executeAction(Action *act) {
    int gr = act->getNr();

    if (gr == 0) {/* gripping */
        if (env[1] == '1')//gripper has some block already
            return 0;
    } else {
        if (env[1] == '0')//gripper is empty (cannot release any block)
            return 0;
    }

    if (gr == 0) {//gripping
        int weight = 0;
        if (env[3] == '1')
            weight = 2;
        if (env[4] == '1')
            weight++;
        if (weight > GRIPPER_MAX_WEIGHT) {
            doReset = 1;
            return 0;
        }
        env[0] = '0';
        env[1] = '1';
        return 0;
    } else {//releasing
        if (env[0] == '0') {
            env[0] = '1';
            env[1] = '0';
            doReset = 1;
            return GRIPPER_REWARD;
        } else {
            return 0;
        }
    }
    return 0; //default return
}


/**
 * Returns an array of all possible actions in this blocks world.
 */
Action **GripperEnvironment::getActions() {
    Action **act = new Action *[2];

    act[0] = new Action(0);
    act[1] = new Action(1);

    return act;
}

/**
 * Converts an action number to an action string (for output purposes).
 */
char *GripperEnvironment::getActionString(Action *act) {
    char *acode = new char[2];
    acode[1] = '\0';
    int nr = act->getNr();
    if (nr == 0)
        acode[0] = 'g';
    else
        acode[0] = 'r';

    return acode;
}

/**
 * Returns the number of possible actions in the environment.
 */
int GripperEnvironment::getNoActions() {
    return 2;
}

/**
 * A reset takes place if all blocks are situated on first stack.
 */
int GripperEnvironment::isReset() {
    return doReset;
}

/**
 * A reset redistributes all blocks randomly.
 */
int GripperEnvironment::reset() {
    int i;
    env[0] = '1';
    env[1] = '0';
    for (i = 2; i < 5; i++)
        if (frand() < 0.5)
            env[i] = '0';
        else
            env[i] = '1';
    doReset = 0;
    env[5] = '\0';
    return 1;
}

/**
 * Sets the environment to test mode.
 */
void GripperEnvironment::doTesting() {
    //not used
}

/**
 * Specifies a test by specifying a perception action resulting perception triple.
 * Returns if another test was generated.
 */
int GripperEnvironment::getNextTest(Perception *p0, Action *act, Perception *p1) {
    return 0;
    //not used
}


/**
 * Resets the environment to normal model.
 */
void GripperEnvironment::endTesting()//Used for testing purposes
{
    //not used
}


ostream &operator<<(ostream &out, GripperEnvironment *env) {
    out << "   |  " << endl;
    out << " -----" << endl;
    if (env->env[1] == '1') {
        if (env->env[2] == '0') {
            out << " |OOO|" << endl;
            out << " |OOO|" << endl;
        } else {
            out << " |###|" << endl;
            out << " |###|" << endl;
        }
        out << "     " << endl;
        out << "     " << endl;
    } else {
        out << " |   |" << endl;
        out << " |   |" << endl;
        if (env->env[2] == '0') {
            out << "  OOO " << endl;
            out << "  OOO " << endl;
        } else {
            out << "  ### " << endl;
            out << "  ### " << endl;
        }
    }
    out << "WWWWWWW" << endl;
    return out;
}
