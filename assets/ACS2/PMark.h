/*
/       (ACS with GA and PEEs inn C++)
/	------------------------------------
/	the Anticipatory Classifier System (ACS) with ALP in action set, GA generalization and PEEs
/
/     (c) by Martin Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 11-30-2000
/
/     classifier mark class.
*/

#ifndef _PMark_h_
#define _PMark_h_

#include<fstream>
#include<math.h>
#include<stdlib.h>

#include"Perception.h"
#include"Condition.h"
#include"CharCharPosList.h"

using namespace std;

class PMark
{
 public:
  PMark(){list=new CharCharPosList();empty=1;}
  PMark(PMark *mark);
  ~PMark() {delete list;}

  int setMark(Condition *con, Perception *percept);

  Condition *getDifferences(Perception *percept);
  
  int isEmpty();
  int isEnhanced();
  int isEqual(PMark *m2, Perception *p0);
  int doesMatch(PMark *m2);
  int doesMatch(Perception *p0);

  friend ostream& operator<<(ostream& out, PMark *pm);

 private:
  int setMark(Perception *percept);
  CharCharPosList *list;
  int empty;
};

#endif

#ifndef _frand_
#define _frand_
#define frand() ((double) rand() / (RAND_MAX+1.0)) 
#endif

