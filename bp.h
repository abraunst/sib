// This file is part of sibilla : inference in epidemics with Belief Propagation
// Author: Alfredo Braunstein


#ifndef BP_H
#define BP_H

#include "factorgraph.h"

template<class T>
struct ArrayType : public std::vector<T>
{
	ArrayType(size_t qj, T const & val) : std::vector<T>(qj*qj, val), qj(qj) {}
	ArrayType(size_t qj) : std::vector<T>(qj*qj), qj(qj) {}
	void clear() { for (int i = 0; i < int(std::vector<T>::size()); ++i) std::vector<T>::operator[](i)*=0.0; }
	size_t dim() const { return qj;}
	inline T & operator()(int sji, int sij) { return std::vector<T>::operator[](qj * sij + sji); }
	inline T const & operator()(int sji, int sij) const { return std::vector<T>::operator[](qj * sij + sji); }
	size_t qj;
};

template<class T>
std::ostream & operator<<(std::ostream & o, ArrayType<T> const & msg)
{
	for (size_t sij = 0; sij < msg.qj; ++sij) {
		for (size_t sji = 0; sji < msg.qj; ++sji) {
			o << msg(sij, sji) << " ";
		}
		o << std::endl;
	}
	return o;
}


typedef ArrayType<real_t> BPMes;




BPMes & operator++(BPMes &);

BPMes & operator--(BPMes &);

typedef FactorGraph<BPMes> BPGraph;
typedef NodeType<BPMes> BPNode;


#endif
