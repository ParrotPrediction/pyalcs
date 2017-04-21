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
/     perception class header
*/


#ifndef _Perception_
#define _Perception_

#include<iostream>
#include<fstream>

using namespace std;

class Perception
{
 public:
  static int length;

  Perception() {percept=new char[length+1]; percept[length]='\0';}
  Perception(Perception *old);
  ~Perception() { delete[] percept; }
  
  void setPerception(char *in);
  void setPerception(Perception *percept);
  char getAttribute(int pos) {if(pos < length && pos > -1) return percept[pos]; else return '\0';}
  void setAttribute(char ch, int pos) {if(pos<length && pos > -1) percept[pos]=ch;}

  int isEqual(Perception *perception);

  friend ostream& operator<<(ostream& out, Perception *p);

 private:
  char* percept;
};

#endif
