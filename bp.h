// This file is part of sibilla : inference in epidemics with Belief Propagation
// Author: Alfredo Braunstein
// Author: Alessandro Ingrosso
// Author: Anna Paola Muntoni

#include <vector>
#include <iostream>
#include <memory>
#include <omp.h>

#include "params.h"


#ifndef FACTORGRAPH_H
#define FACTORGRAPH_H

extern int const Tinf;


template<class T>
struct Message : public std::vector<T>
{
	Message(size_t qj, T const & val) : std::vector<T>(qj*qj, val), qj(qj) {}
	Message(size_t qj) : std::vector<T>(qj*qj), qj(qj) {}
	void clear() { for (int i = 0; i < int(std::vector<T>::size()); ++i) std::vector<T>::operator[](i)*=0.0; }
	size_t dim() const { return qj;}
	inline T & operator()(int sji, int sij) { return std::vector<T>::operator[](qj * sij + sji); }
	inline T const & operator()(int sji, int sij) const { return std::vector<T>::operator[](qj * sij + sji); }
	size_t qj;
};

typedef Message<real_t> BPMes;

template<class Mes>
struct NeighType {
	NeighType(int index, int pos) : index(index), pos(pos), t(1, Tinf), lambdas(1, 0.0), msg(1, 1.0) {
		omp_init_lock(&lock_);

	}
	int index;  // index of the node
	int pos;    // position of the node in neighbors list
	std::vector<int> t; // time index of contacts
	std::vector<real_t> lambdas; // transmission probability
	BPMes msg; // BP msg nij^2 or
	void lock() const { omp_set_lock(&lock_); }
	void unlock() const { omp_unset_lock(&lock_); }
	mutable omp_lock_t lock_;
};

template<class TMes>
struct NodeType {
	NodeType(std::shared_ptr<Proba> prob_i, std::shared_ptr<Proba> prob_r, int index) :
		prob_i(prob_i),
		prob_r(prob_r),
		prob_i0(prob_i),
		prob_r0(prob_r),
		f_(0),
		df_i(RealParams(0.0, prob_i->theta.size())),
		df_r(RealParams(0.0, prob_r->theta.size())),
		index(index)
	{
		times.push_back(-1);
		times.push_back(Tinf);
		for (int t = 0; t < 2; ++t) {
			bt.push_back(1);
			ht.push_back(1);
			bg.push_back(1);
			hg.push_back(1);
		}

	}
	void push_back_time(times_t t) {
		times.back() = t;
		times.push_back(Tinf);
                ht.push_back(ht.back());
                hg.push_back(hg.back());
                bt.push_back(bt.back());
                bg.push_back(bg.back());
	}
	std::shared_ptr<Proba> prob_i;
	std::shared_ptr<Proba> prob_r;
	std::shared_ptr<Proba> prob_i0;
	std::shared_ptr<Proba> prob_r0;
	std::vector<times_t> times;
	std::vector<real_t> bt;  // marginals infection times T[ni+2]
	std::vector<real_t> bg;  // marginals recovery times G[ni+2]
	std::vector<real_t> ht;  // message infection times T[ni+2]
	std::vector<real_t> hg;  // message recovery times G[ni+2]
	std::vector<NeighType<TMes>> neighs;	   // list of neighbors
	real_t f_;
	real_t err_;
	RealParams df_i;
	RealParams df_r;
	int index;
};

template<class TMes>
class FactorGraph {
public:
	typedef TMes Mes;
	typedef NodeType<Mes> Node;
	typedef NeighType<Mes> Neigh;
	std::vector<Node> nodes;
	FactorGraph(Params const & params,
		std::vector<std::tuple<int,int,times_t,real_t> > const & contacts,
		std::vector<std::tuple<int,int,times_t> > const & obs,
		std::vector<std::tuple<int, std::shared_ptr<Proba>, std::shared_ptr<Proba>, std::shared_ptr<Proba>, std::shared_ptr<Proba>> > const & individuals = std::vector<std::tuple<int, std::shared_ptr<Proba>, std::shared_ptr<Proba>, std::shared_ptr<Proba>, std::shared_ptr<Proba>>>());
	int find_neighbor(int i, int j) const;
	void append_contact(int i, int j, times_t t, real_t lambdaij, real_t lambdaji = DO_NOT_OVERWRITE);
	void drop_contacts(times_t t);
	void append_observation(int i, int s, times_t t);
	void append_time(int i, times_t t);
	void add_node(int i);
	void init();
	void set_fields(int i, std::vector<int> const & sobs, std::vector<times_t> const & tobs);
	void set_field(int i, int s, int t);
	void reset_observations(std::vector<std::tuple<int, int, times_t> > const & obs);
	real_t update(int i, real_t damping, bool learn = false);
	void show_graph();
	void show_beliefs(std::ostream &);
	real_t iterate(int maxit, real_t tol, real_t damping, bool learn = false);
	real_t iteration(real_t damping, bool learn = false);
	real_t loglikelihood() const;
	void show_msg(std::ostream &);
	Params params;
	enum ARRAY_ENUM { DO_NOT_OVERWRITE = -1 };
};

template<class TMes>
std::ostream & operator<<(std::ostream &, FactorGraph<TMes> const &);

typedef FactorGraph<BPMes> BPGraph;


template<class TMes>
void FactorGraph<TMes>::add_node(int i)
{
	for (int j = nodes.size(); j < i + 1; ++j)
		nodes.push_back(Node(params.prob_i, params.prob_r, j));
}

template<class TMes>
void FactorGraph<TMes>::show_graph()
{
	std::cerr << "Number of nodes " <<  int(nodes.size()) << std::endl;
	for(int i = 0; i < int(nodes.size()); i++) {
		std::cerr << "### index " << i << "###" << std::endl;
		std::cerr << "### in contact with " <<  int(nodes[i].neighs.size()) << "nodes" << std::endl;
		std::vector<Neigh> const & aux = nodes[i].neighs;
		for (int j = 0; j < int(aux.size()); j++) {
			std::cerr << "# neighbor " << aux[j].index << std::endl;
			std::cerr << "# in position " << aux[j].pos << std::endl;
			std::cerr << "# in contact " << int(aux[j].t.size()) << " times, in t: ";
			for (int s = 0; s < int(aux[j].t.size()); s++)
				std::cerr << aux[j].t[s] << " ";
			std::cerr << " " << std::endl;
		}
	}
}

template<class TMes>
void FactorGraph<TMes>::show_beliefs(std::ostream & ofs)
{
	for(int i = 0; i < int(nodes.size()); ++i) {
		Node & f = nodes[i];
		ofs << "node " << i << ":" << std::endl;
		for (int t = 0; t < int(f.bt.size()); ++t) {
			ofs << "    " << f.times[t] << " " << f.bt[t] << " (" << f.ht[t] << ") " << f.bg[t] << " (" << f.hg[t] << ")" << std::endl;
		}
	}

}

template<class TMes>
void FactorGraph<TMes>::show_msg(std::ostream & o)
{
	for(int i = 0; i < int(nodes.size()); ++i) {
		auto & n = nodes[i];
		for(int j = 0; j < int(n.neighs.size()); ++j) {
			auto & v = n.neighs[j];
			o << i << " <- " << v.index << " : " << std::endl;
			for (int sij = 0; sij < int(v.msg.qj); ++sij) {
				for (int sji = 0; sji < int(v.msg.qj); ++sji) {
					o << v.msg(sij, sji) << " ";
				}
				o << std::endl;
			}

		}
	}
}

template<class TMes>
real_t FactorGraph<TMes>::iteration(real_t damping, bool learn)
{
	int const N = nodes.size();
	real_t err = 0.0;
	std::vector<int> perm(N);
	for(int i = 0; i < N; ++i)
		perm[i] = i;
	random_shuffle(perm.begin(), perm.end());
#pragma omp parallel for reduction(max:err)
	for(int i = 0; i < N; ++i)
		err = std::max(err, update(perm[i], damping, learn));
	return err;
}

template<class TMes>
real_t FactorGraph<TMes>::iterate(int maxit, real_t tol, real_t damping, bool learn)
{
	real_t err = std::numeric_limits<real_t>::infinity();
	for (int it = 1; it <= maxit; ++it) {
		err = iteration(damping, learn);
		std::cout << "it: " << it << " err: " << err << std::endl;
		if (err < tol)
			break;
	}
	return err;
}

template<class TMes>
void drop_time(FactorGraph<TMes> & fg, int t)
{
        fg.drop_contacts(t);
        int n = fg.nodes.size();
        for (int i = 0; i < n; ++i) {
                NodeType<TMes> & f = fg.nodes[i];
                if (t == f.times[1]) {
                        f.bt.erase(f.bt.begin());
                        f.bg.erase(f.bg.begin());
                        f.ht.erase(f.ht.begin());
                        f.hg.erase(f.hg.begin());
			f.times.erase(f.times.begin() + 1);
			int m = f.neighs.size();
			for (int j = 0; j < m; ++j) {
				NeighType<TMes> & v = f.neighs[j];
				for (int k = 0; k < int(v.t.size()); ++k) {
					--v.t[k];
				}
			}
                }
		f.times[0] = t;
        }
}

#endif
