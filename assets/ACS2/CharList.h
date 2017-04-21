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
/     list structure of one attribute in the mark header
*/


#ifndef _CharList_h_
#define _CharList_h_

#include<iostream>
#include<fstream>

using namespace std;

class CharList
{
 public:
  CharList(char c) {this->c=c; next=0;}
  CharList(CharList *oldList);
  ~CharList() {delete next;}

  int insert(char c);
  int remove(char c);
  int doesContain(char c);
  int isIdentical(CharList *list);
  int isEnhanced(){if(next!=0) return 1; else return 0;}

  friend ostream& operator<<(ostream& out, CharList *cl);
 private:
  CharList *next;
  char c;
};

#endif
