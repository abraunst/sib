// This file is part of sibilla : inference in epidemics with Belief Propagation
// Author: Alfredo Braunstein
// Author: Alessandro Ingrosso
// Author: Anna Paola Muntoni
// Author: Indaco Biazzo



#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>
#include <algorithm>
#include <assert.h>
#include <tuple>
#include <exception>
#include "mf.h"
#include "cavity.h"

using namespace std;


template<>
char const * MFGraph::name()
{
	return "MFGraph";
}

template<>
char const * MFNode::name()
{
	return "MFNode";
}

std::ostream & operator<<(std::ostream &o, MFMes const & msg)
{
	for (size_t i = 0; i < msg.size(); ++i)
		o << msg << " ";
	return o << std::endl;
}

MFMes & operator++(MFMes & msg)
{
	msg.push_back(0.0);
	return msg;
}

MFMes & operator--(MFMes & msg)
{
	msg.erase(msg.begin(), msg.begin() + 1);
	return msg;
}

template<>
real_t MFGraph::update(int i, real_t damping, bool learn)
{
	return 0.0;
}


template<>
real_t MFGraph::loglikelihood() const
{
	return 0.0;
}

