// This file is part of sibilla : inference in epidemics with Belief Propagation
// Author: Alfredo Braunstein


#ifndef MF_H
#define MF_H

#include "factorgraph.h"


template<class T>
struct VecType : public std::vector<T>
{
	VecType(int n) : std::vector<T>(n) {}
	VecType(int n, T val) : std::vector<T>(n, val) {}
};

typedef VecType<real_t> MFMes;

std::ostream & operator<<(std::ostream &o, MFMes const & msg);


MFMes & operator++(MFMes &);

MFMes & operator--(MFMes &);

typedef FactorGraph<MFMes> MFGraph;
typedef NodeType<MFMes> MFNode;


#endif
