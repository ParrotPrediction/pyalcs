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
/     classifier class.
*/

#include<iostream>
#include<assert.h>

#include"Condition.h"
#include"Action.h"
#include"Effect.h"
#include"PMark.h"
#include"Classifier.h"

/**
 * Constructs new empty classifier with specified action.
 */
Classifier::Classifier(Action *action) {
    initialize(new Condition(), new Action(action), new Effect(), new PMark(), 0, Q_INI, R_INI, IR_INI, AVT_INI);
}

/**
 * Copies complete classifier with all components except for the mark!
 */
Classifier::Classifier(Classifier *cl, int time) {
    initialize(new Condition(cl->C), new Action(cl->A), new Effect(cl->E), new PMark(), time, cl->q, cl->r, cl->i,
               cl->tav);
}

/**
 * Creates a classifier that anticipates the change correctly.
 */
Classifier::Classifier(Perception *p0, Action *act, Perception *p1, int time) {
    E = new Effect();
    C = E->getAndSpecialize(p0, p1);
    initialize(C, new Action(act), E, new PMark(), time, Q_INI, R_INI, IR_INI, AVT_INI);
}

/**
 * Controls the expected case of a classifier.
 */
Classifier *Classifier::expectedCase(Perception *percept, int time) {
    E->updateEnhancedEffectProbs(percept, BETA);

    //Return if the specificity threshold is already reached
    int noSpec = getUnchangeSpec();

    //Check out the possible specializations comparing mark to the percept
    Condition *diff = M->getDifferences(percept);
    if (diff == 0) {
        //No possibility for a further specialization was found
        if (DO_PEES && !M->isEmpty()) {
            ee = 1;
        }
        increaseQuality();
        return 0;
    }

    //Create a new classifier
    Classifier *cl = new Classifier(this, time);

    int noSpecNew = diff->getList()->getSize();

    if (noSpec >= U_MAX) {
        //Remove positions if the classifier is too specific anyways
        while (noSpec >= U_MAX) {
            while (!cl->generalizeRandomUnchangeCond(noSpec));//Remove definitely one of the old specializations
            noSpec--;
        }
        while (noSpec + noSpecNew > U_MAX) {
            if (frand() < 0.5) {
                if (diff->generalize((int) (frand() * noSpecNew)))
                    noSpecNew--;
            } else {
                if (cl->generalizeRandomUnchangeCond(noSpec))
                    noSpec--;
            }
        }
    } else {
        //Remove positions if toxo many specializations are suggested
        while (noSpec + noSpecNew > U_MAX) {
            if (diff->generalize((int) (frand() * noSpecNew)))
                noSpecNew--;
            else
                cout << "Less specialization in expected case should always work!?" << endl;
        }
    }

    if (!cl->C->specialize(diff)) {
        cout << "Mistake in the expected case during specialization!" << endl;
        exit(0);
    }
    delete diff;
    if (cl->q < 0.5)
        cl->q = 0.5;

    return cl;
}

/**
 * Generalizes randomly one unchanging attribute in the condition. 
 * An unchanging attribute is one that is anticipated not to change in the effect part.
 */
int Classifier::generalizeRandomUnchangeCond(int noSpec) {
    int pos = (int) (frand() * noSpec); //determine which unchanging attribute to generalize

    CharPosList *cpl = C->getList();
    cpl->reset();
    ProbCharPosList *epl = E->getList();
    epl->reset();

    CharPosItem *cpi = cpl->getNextItem();
    ProbCharPosItem *epi = epl->getNextItem();
    while (cpi != 0 && epi != 0) {
        if (cpi->getPos() < epi->getPos()) {
            if (pos == 0)
                break;
            pos--;
            cpi = cpl->getNextItem();
        } else if (cpi->getPos() == epi->getPos()) {
            if (epi->getItem()->doesContain(cpi->getChar())) {//explicit equal in enhanced effect part
                if (pos == 0)
                    break;
                pos--;
            }
            cpi = cpl->getNextItem();
            epi = epl->getNextItem();
        } else {
            epi = epl->getNextItem();
        }
    }

    if (cpi != 0) {
        C->generalize(cpi);
        return 1;
    }
    return 0;
}

/**
 * Returns is the Effect part contains specified attributes.
 */
int Classifier::doesAnticipateChange() {
    return E->getSpecificity();
}

/**
 * Determines the number of specified unchanging attributes in the classifier.
 * An unchanging attribute is one that is anticipated not to change in the effect part.
 */
int Classifier::getUnchangeSpec() {
    CharPosList *cpl = C->getList();
    cpl->reset();
    ProbCharPosList *epl = E->getList();
    epl->reset();
    int spec = 0;

    CharPosItem *cpi = cpl->getNextItem();
    ProbCharPosItem *epi = epl->getNextItem();
    while (cpi != 0 && epi != 0) {
        if (cpi->getPos() < epi->getPos()) {
            spec++;
            cpi = cpl->getNextItem();
        } else if (cpi->getPos() == epi->getPos()) {
            if (epi->getItem()->doesContain(cpi->getChar()))//explicit equal in enhanced effect part
                spec++;
            cpi = cpl->getNextItem();
            epi = epl->getNextItem();
        } else {
            epi = epl->getNextItem();
        }
    }
    while (cpi != 0) {
        spec++;
        cpi = cpl->getNextItem();
    }

    return spec;
}

/**
 * Controls the unexpected case of a classifier.
 * Returns a specialized classifier if generation was possible, 0 otherwise.
 */
Classifier *Classifier::unexpectedCase(Perception *p0, Perception *p1, int time) {
    decreaseQuality();
    mark(p0);

    if (E->isSpecializable(p0, p1) != 0) {
        Classifier *cl = new Classifier(this, time);

        Condition *diff = cl->E->getAndSpecialize(p0, p1);
        cl->C->specialize(diff);

        delete diff;

        if (cl->q < 0.5)
            cl->q = 0.5;

        return cl;
    }

    return 0;
}

/**
 * Returns if the classifier is equal to classifier cl in condition, action and effect part.
 */
int Classifier::isSimilar(Classifier *cl) {
    if (C->isEqual(cl->C) && A->isEqual(cl->A) && E->isEqual(cl->E))
        return 1;
    return 0;
}

/**
 * Returns if the classifier subsumes classifier cl.
 */
int Classifier::doesSubsume(Classifier *cl) {
    if (isSubsumer() && isMoreGeneral(cl) && C->doesMatch(cl->C) && A->isEqual(cl->A) && E->isEqual(cl->E))
        return 1;
    return 0;
}

/**
 * Returns if the classifier matches the percept.
 */
int Classifier::doesMatch(Perception *percept) {
    return C->doesMatch(percept);
}

/**
 * Returns if the classifier correctly anticipates the change from p0 to p1.
 */
int Classifier::doesAnticipateCorrect(Perception *p0, Perception *p1) {
    return E->doesAnticipateCorrectly(p0, p1);
}

/**
 * Returns the anticipation, the classifier believes to happen most probably.
 * This is usually the normal anticipation. However, if PEEs are activated, the most probable
 * value of each attribute is returned.
 */
Perception *Classifier::getBestAnticipation(Perception *percept) {
    return E->getBestAnticipation(percept);
}

/**
 * Returns the backwards anticipation.
 * Returns 0 if the backwards anticipation was impossible to create! This is the case if
 * changing attributes are not specified in the conditions.
 */
Perception *Classifier::getBackwardsAnticipation(Perception *percept) {
    Perception *p = C->getBackwardsAnticipation(percept);
    if (!E->doesSpecifyOnlyChangesBackwards(p, percept)) {
        //If a specified attribute in the effect part matches the anticipated p, the backwards anticipation fails
        // (because a specified attribue in E means a change!).
        delete p;
        return 0;
    }
    return p;
}

/**
 * Returns if 'percept' is matched by the anticipations.
 * This is only the case if the specified conditions that have '#'-symbols in the effect part are also matched!
 */
int Classifier::doesMatchBackwards(Perception *percept) {
    Perception *p = C->getBackwardsAnticipation(percept);
    if (E->doesMatch(percept, p)) {
        delete p;
        return 1;
    }
    delete p;
    return 0;
}

int Classifier::doesLink(Classifier *cl) {
    CharPosList *cpl = C->getList();
    ProbCharPosList *pcpl = E->getList();
    CharPosList *cpl2 = cl->C->getList();
    ProbCharPosList *pcpl2 = cl->E->getList();

    cpl->reset();
    cpl2->reset();
    pcpl->reset();
    pcpl2->reset();

    //Check if cl.C matches this classifier
    CharPosItem *cpi = cpl->getNextItem();
    ProbCharPosItem *pcpi = pcpl->getNextItem();
    CharPosItem *cpi2 = cpl2->getNextItem();
    ProbCharPosItem *pcpi2 = pcpl2->getNextItem();

    while ((cpi2 != 0 || pcpi2 != 0) && (cpi != 0 || pcpi != 0)) {
        if (cpi != 0) {
            if (pcpi == 0 || cpi->getPos() < pcpi->getPos()) {
                if (!doesLink(cpi->getChar(), cpi->getPos(), cpl2, &cpi2, pcpl2, &pcpi2))
                    return 0;
                cpi = cpl->getNextItem();//this item is OK -> get next one
            } else {//given: pcpi->getPos() is the next existent, relevant item
                if (!doesTightLink(pcpi->getItem()->getBestChar(), pcpi->getPos(), cpl2, &cpi2))
                    return 0;
                if (cpi->getPos() == pcpi->getPos())//E overrules C so get next C item as well
                    cpi = cpl->getNextItem();
                pcpi = pcpl->getNextItem();//this item is OK -> get next one
            }
        } else {
            if (!doesTightLink(pcpi->getItem()->getBestChar(), pcpi->getPos(), cpl2, &cpi2))
                return 0;
            pcpi = pcpl->getNextItem();//this item is OK -> get next one
        }
    }
    return 1;
}


int Classifier::doesLink(char chr, int pos, CharPosList *cpl2, CharPosItem **cpi2, ProbCharPosList *pcpl2,
                         ProbCharPosItem **pcpi2) {
    while ((*cpi2) != 0 && (*cpi2)->getPos() < pos)//look for next relevant att. in cl.C
        (*cpi2) = cpl2->getNextItem();
    if ((*cpi2) == 0 || (*cpi2)->getPos() > pos) {//next specific att. in cl.C comes later
        while ((*pcpi2) != 0 && (*pcpi2)->getPos() < pos)//look for next relevant specific att. in cl.E
            (*pcpi2) = pcpl2->getNextItem();
        if ((*pcpi2) != 0 && (*pcpi2)->getPos() == pos) {
            if ((*pcpi2)->getItem()->getBestChar() == chr)//anticpates no change -> cannot link
                return 0;
        }
    } else {//given: pos==(*cpi2)->getPos()
        if ((*cpi2)->getChar() != chr) // does not match -> cannot link
            return 0;
    }
    return 1;
}


int Classifier::doesTightLink(char chr, int pos, CharPosList *cpl2, CharPosItem **cpi2) {
    while ((*cpi2) != 0 && (*cpi2)->getPos() < pos)//look for next relevant att. in cl.C
        (*cpi2) = cpl2->getNextItem();
    if ((*cpi2) == 0 || (*cpi2)->getPos() > pos) {//no tight link since condition does not specify required position
        return 0;
    } else {//given: pos==(*cpi2)->getPos()
        if ((*cpi2)->getChar() != chr) // does not match -> cannot link
            return 0;
    }
    return 1;
}


/**
 * Returns a new classifier that merges this one and cl2. 
 * Conditions are specialized in all attributes that are specialized in this condition or the condition of cl2.
 * Effects are merged.
 * The mark of the new classifier stays empty.
 */
Classifier *Classifier::mergeClassifiers(Classifier *cl2, Perception *percept, int time) {
    Classifier *newCl = new Classifier();

    newCl->C = new Condition(C);
    newCl->C->specialize(cl2->C);

    newCl->A = new Action(A);

    newCl->E = new Effect(E, cl2->E, q, cl2->q, percept);

    newCl->M = new PMark();

    newCl->r = (r + cl2->r) / 2.;
    newCl->q = (q + cl2->q) / 2.;
    if (newCl->q < Q_INI)
        newCl->q = Q_INI;
    newCl->num = 1;
    newCl->tga = time;
    newCl->talp = time;
    newCl->tav = 0;
    newCl->exp = 1;
    newCl->ee = 0;

    return newCl;
}


/**
 * Executes the generalizing mutation in the classifier.
 */
void Classifier::mutate() {
    int size = C->getSpecificity();
    for (int i = 0; i < size; i++) {
        if (frand() < MU) {
            assert(C->generalize(i) == 1);
            i--;
            size--;
        }
    }
}

/**
 * Executes crossover (chooses the crossover type).
 * Note that this method actually executes crossover. If crossover should be applied is checked elsewhere.
 */
void Classifier::crossover(Classifier *cl2) {
    switch (X_TYPE) {
        case 0:
            C->uniformCrossover(cl2->C);
            break;
        case 1:
            C->onePointCrossover(cl2->C, Perception::length);
            break;
        case 2:
            C->twoPointCrossover(cl2->C, Perception::length);
            break;
        default:
            cout << "X_TYPE is not set correctly!!!" << endl;
            return;
    }
    q = (q + cl2->q) / 2.;
    r = (r + cl2->r) / 2.;
    cl2->q = q;
    cl2->r = r;
}

/**
 * Creates a classifier with specified condition, action, and effect.
 */
Classifier::Classifier(Condition *con, Action *act, Effect *eff) {
    initialize(con, act, eff, new PMark(), 0, Q_INI, R_INI, IR_INI, AVT_INI);
}

/**
 * Creates a classifier with specified condition, action, effect, and time stamps.
 */
Classifier::Classifier(Condition *con, Action *act, Effect *eff, int time) {
    initialize(con, act, eff, new PMark(), time, Q_INI, R_INI, IR_INI, AVT_INI);
}

/**
 * Creates an even more initialized classifier.
 */
Classifier::Classifier(Condition *con, Action *act, Effect *eff, int time, double quality, double rewardPrediction,
                       double imRePrediction) {
    initialize(con, act, eff, new PMark(), time, quality, rewardPrediction, imRePrediction, AVT_INI);
}

/**
 * Initializes and sets the specified classifier parameters.
 */
void Classifier::initialize(Condition *con, Action *act, Effect *eff, PMark *mark, int time, double qual, double rew,
                            double imRew, double avApplTime) {
    C = con;
    A = act;
    E = eff;
    M = mark;
    q = qual;
    r = rew;
    i = imRew;
    num = 1;
    tga = time;
    talp = time;
    tav = avApplTime;
    exp = 1;
    ee = 0;
}

/**
 * Sets the ALP time stamp and the application average parameter.
 */
double Classifier::setALPTimeStamp(int time) {
    if (1. / exp > BETA)
        tav = (tav * exp + (time - talp)) /
              (exp + 1);
    else
        tav += BETA * ((time - talp) - tav);
    talp = time;
    return tav;
}

/**
 * Controls if the classifier satisfies the subsume criteria.
 */
int Classifier::isSubsumer() {
    if (exp > THETA_EXP)
        if (q > THETA_R)
            if (M->isEmpty())
                return 1;
    return 0;
}

/**
 * Returns if the classifier is marked.
 */
int Classifier::isMarked() {
    if (M->isEmpty())
        return 0;
    return 1;
}

/**
 * Returns if the classifier is formally more general than classifier 'cl'.
 */
int Classifier::isMoreGeneral(Classifier *cl) {
    if (C->getSpecificity() < cl->C->getSpecificity())
        return 1;
    return 0;
}


ostream &operator<<(ostream &out, Classifier *cl) {
    out << cl->C << " " << cl->A << " " << cl->E << " ";
    out << cl->M << " ";
    out << cl->q << "\t" << cl->r << "\t" << cl->i << "\t" << cl->exp << " " << cl->tga << "\t" << cl->talp << " "
        << cl->tav << " " << cl->num;
    return out;
}
