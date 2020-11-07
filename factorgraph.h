// This file is part of sibilla : inference in epidemics with Belief Propagation
// Author: Alfredo Braunstein
// Author: Alessandro Ingrosso
// Author: Anna Paola Muntoni

#ifndef FACTORGRAPH_H
#define FACTORGRAPH_H

#include <vector>
#include <iostream>
#include <memory>
#include <omp.h>

#include "params.h"



extern int const Tinf;


template<class TMes>
struct NeighType {
	NeighType(int index, int pos) : index(index), pos(pos), t(1, Tinf), lambdas(1, 0.0), msg(1, 1.0) {
		omp_init_lock(&lock_);

	}
	int index;  // index of the node
	int pos;    // position of the node in neighbors list
	std::vector<int> t; // time index of contacts
	std::vector<real_t> lambdas; // transmission probability
	TMes msg; // BP msg nij^2 or
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
void FactorGraph<TMes>::append_contact(int i, int j, times_t t, real_t lambdaij, real_t lambdaji)
{
	if (i == j)
		throw std::invalid_argument("self loops are not allowed");
        add_node(i);
        add_node(j);
	Node & fi = nodes[i];
	Node & fj = nodes[j];
	int qi = fi.times.size();
	int qj = fj.times.size();
	if (fi.times[qi - 2] > t || fj.times[qj - 2] > t)
		throw std::invalid_argument("time of contacts should be ordered");

	int ki = find_neighbor(i, j);
	int kj = find_neighbor(j, i);

	if (ki == int(fi.neighs.size())) {
		assert(kj == int(fj.neighs.size()));
		fi.neighs.push_back(Neigh(j, kj));
		fj.neighs.push_back(Neigh(i, ki));
	}

	Neigh & ni = fi.neighs[ki];
	Neigh & nj = fj.neighs[kj];
	if (fi.times[qi - 2] < t) {
		fi.push_back_time(t);
                ++qi;
	}
	if (fj.times[qj - 2] < t) {
		fj.push_back_time(t);
                ++qj;
	}
	if (ni.t.size() < 2 || ni.t[ni.t.size() - 2] < qi - 2) {
		ni.t.back() = qi - 2;
		nj.t.back() = qj - 2;
		ni.t.push_back(qi - 1);
		nj.t.push_back(qj - 1);
		if (lambdaij != DO_NOT_OVERWRITE)
			ni.lambdas.back() = lambdaij;
		if (lambdaji != DO_NOT_OVERWRITE)
			nj.lambdas.back() = lambdaji;
                ni.lambdas.push_back(0.0);
                nj.lambdas.push_back(0.0);
		++ni.msg;
		++nj.msg;
	} else if (ni.t[ni.t.size() - 2] == qi - 2) {
		if (lambdaij != DO_NOT_OVERWRITE)
			ni.lambdas[ni.t.size() - 2] = lambdaij;
		if (lambdaji != DO_NOT_OVERWRITE)
			nj.lambdas[nj.t.size() - 2] = lambdaji;
	} else {
		throw std::invalid_argument("time of contacts should be ordered");
	}
        // adjust infinite times
        for (int k = 0; k < int(fi.neighs.size()); ++k) {
                fi.neighs[k].t.back() = qi - 1;
	}
        for (int k = 0; k < int(fj.neighs.size()); ++k) {
                fj.neighs[k].t.back() = qj - 1;
	}
}


template<class TMes>
FactorGraph<TMes>::FactorGraph(Params const & params,
		std::vector<std::tuple<int, int, times_t, real_t> > const & contacts,
		std::vector<std::tuple<int, int, times_t> > const & obs,
		std::vector<std::tuple<int, std::shared_ptr<Proba>, std::shared_ptr<Proba>, std::shared_ptr<Proba>, std::shared_ptr<Proba>>> const & individuals) :
	params(params)
{
	for (auto it = individuals.begin(); it != individuals.end(); ++it) {
		if (!std::get<1>(*it) || !std::get<1>(*it) || !std::get<1>(*it)|| !std::get<1>(*it))
			throw std::invalid_argument("invalid individual definition");
		add_node(std::get<0>(*it));
		Node & n = nodes[std::get<0>(*it)];
		n.prob_i = std::get<1>(*it);
		n.prob_r = std::get<2>(*it);
		n.prob_i0 = std::get<3>(*it);
		n.prob_r0 = std::get<4>(*it);
		n.df_i = RealParams(n.prob_i->theta.size());
		n.df_r = RealParams(n.prob_r->theta.size());
	}
	auto ic = contacts.begin(), ec = contacts.end();
	auto io = obs.begin(), eo = obs.end();
	while (ic != ec || io != eo) {
		int tc = ic == ec ? Tinf : std::get<2>(*ic);
		int to = io == eo ? Tinf : std::get<2>(*io);
		if (tc < to) {
			// cerr << "appending contact" << get<0>(*ic) << " " <<  get<1>(*ic)<< " " <<  get<2>(*ic) << " " <<  get<3>(*ic) << endl;
			append_contact(std::get<0>(*ic), std::get<1>(*ic), std::get<2>(*ic), std::get<3>(*ic));
			ic++;
		} else {
			// cerr << "appending obs" << get<0>(*io) << " " <<  get<1>(*io)<< " " <<  get<2>(*io)  << endl;
			append_time(std::get<0>(*io), std::get<2>(*io));
			io++;
		}
	}
	reset_observations(obs);
}

template<class TMes>
int FactorGraph<TMes>::find_neighbor(int i, int j) const
{
	int k = 0;
	for (; k < int(nodes[i].neighs.size()); ++k)
		if (j == nodes[i].neighs[k].index)
			break;
	return k;
}

template<class TMes>
void norm_msg(TMes & msg)
{
	real_t S = 0;
	for(int n = 0; n < int(msg.size()); ++n)
		S += msg[n];
	if (!(S > 0))
		throw std::domain_error("singularity error");
	for(int n = 0; n < int(msg.size()); ++n)
		msg[n] /= S;
}

template<class TMes>
real_t setmes(TMes & from, TMes & to, real_t damp)
{
	int n = from.size();
	real_t s = 0;
	for (int i = 0; i < n; ++i) {
		s += from[i];
	}
	real_t err = 0;
	for (int i = 0; i < n; ++i) {
		if (!(s > 0)){
			from[i] = 1./n;
			err = std::numeric_limits<real_t>::infinity();
		} else {
			from[i] /= s;
			err = std::max(err, std::abs(from[i] - to[i]));
		}
		to[i] = damp*to[i] + (1-damp)*from[i];
	}
	return err;
}


template<class TMes>
std::ostream & operator<<(std::ostream & ost, FactorGraph<TMes> const & f)
{
	int nasym = 0;
	int nedge = 0;
	int ncont = 0;
	for(int i = 0; i < int(f.nodes.size()); ++i) {
		for (auto vit = f.nodes[i].neighs.begin(), vend = f.nodes[i].neighs.end(); vit != vend; ++vit) {
                        if (vit->index < i)
                                continue;
			++nedge;
			ncont += vit->lambdas.size() - 1;
			if (vit->lambdas != f.nodes[vit->index].neighs[vit->pos].lambdas)
				++nasym;
		}
	}

	return ost << "FactorGraph\n"
                << "            nodes: " << f.nodes.size() << "\n"
		<< "            edges: " << nedge << " ("  << nasym <<  " asymmetric)\n"
		<< "    time contacts: " << ncont;
}

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

template<class TMes>
void FactorGraph<TMes>::set_field(int i, int s, int tobs)
{
	Node & n = nodes[i];
        int qi = n.times.size();
        switch (s) {
                case 0:
			for (int t = 0; t < qi; ++t)
				n.ht[t] *= params.fn_rate * (n.times[t] < tobs) + (1 - params.fn_rate) * (n.times[t] >= tobs);
                        break;
                case 1:
			for (int t = 0; t < qi; ++t) {
				n.ht[t] *= (1 - params.fp_rate) * (n.times[t] < tobs) + params.fp_rate * (n.times[t] >= tobs);
				n.hg[t] *= (n.times[t] >= tobs);
			}
                        break;
                case 2:
			for (int t = 0; t < qi; ++t) {
				n.ht[t] *= (n.times[t] < tobs);
				n.hg[t] *= (n.times[t] < tobs);
			}
                        break;
        }
}

template<class TMes>
void FactorGraph<TMes>::append_time(int i, times_t t)
{
	add_node(i);
	Node & n = nodes[i];
	// most common case
	if (t == n.times[n.times.size() - 2]
		|| t == *lower_bound(n.times.begin(), n.times.end(), t))
		return;
	if (t > n.times[n.times.size() - 2]) {
		n.push_back_time(t);
                // adjust infinite times
                for (int j = 0; j < int(n.neighs.size()); ++j) {
                        n.neighs[j].t.back() = n.times.size() - 1;
                }
		return;
        }
	std::cerr << t << " < " << n.times[n.times.size() - 2] << std::endl;
	throw std::invalid_argument("observation time unexistent and too small");
}

template<class TMes>
void FactorGraph<TMes>::append_observation(int i, int s, times_t t)
{
	append_time(i, t);
	set_field(i, s, t);
}

template<class TMes>
void FactorGraph<TMes>::reset_observations(std::vector<std::tuple<int, int, times_t> > const & obs)
{
	std::vector<std::vector<times_t>> tobs(nodes.size());
	std::vector<std::vector<int>> sobs(nodes.size());
	for (auto it = obs.begin(); it != obs.end(); ++it) {
		sobs[std::get<0>(*it)].push_back(std::get<1>(*it));
		tobs[std::get<0>(*it)].push_back(std::get<2>(*it));
	}
	int largeT = 0;
	for (int i = 0; i < int(nodes.size()); ++i) {
		largeT = std::max(largeT, int(nodes[i].times.size()));
	}
	std::vector<int> FS(largeT), FI(largeT), TS(largeT), TI(largeT), TR(largeT);
	std::vector<real_t> pFS(largeT, 1.0), pFI(largeT, 1.0), pTS(largeT, 1.0), pTI(largeT, 1.0);

	for (int t = 1; t < largeT; ++t) {
		pTI[t] = pTI[t-1] * (1-params.fp_rate);
		pFI[t] = pFI[t-1] * params.fp_rate;
		pTS[t] = pTS[t-1] * (1-params.fn_rate);
		pFS[t] = pFS[t-1] * params.fn_rate;
	}
	for (int i = 0; i < int(nodes.size()); ++i) {
		fill(TS.begin(), TS.end(), 0);
		fill(FS.begin(), FS.end(), 0);
		fill(TI.begin(), TI.end(), 0);
		fill(FI.begin(), FI.end(), 0);
		fill(TR.begin(), TR.end(), 0);
		// this assumes ordered observation times
		int T = nodes[i].times.size();
		int t = 0;
		for (int k = 0; k < int(tobs[i].size()); ++k) {
			int state = sobs[i][k];
			int to = tobs[i][k];
			while (nodes[i].times[t] != to && t < T)
				t++;
			if (nodes[i].times[t] != to)
				throw std::invalid_argument(("this is a bad time: node" + std::to_string(i) + " time " + std::to_string(t)).c_str());
			switch(state) {
				case 0:
					FS[0]++;
					FS[t]--;
					TS[t]++;
					break;
				case 1:
					TI[0]++;
					TI[t]--;
					FI[t]++;
					TR[0]++;
					TR[t]--;
					break;
				case 2:
					TR[t]++;
					TI[t]++;
					break;
			}
		}
		int fs = 0, fi = 0, ts = 0, ti = 0, tr = 0;
		for (int t = 0; t < T; ++t) {
			fs += FS[t];
			fi += FI[t];
			ts += TS[t];
			ti += TI[t];
			tr += TR[t];
			nodes[i].ht[t] = pFS[fs] * pTS[ts] * pFI[fi] * pTI[ti];
			nodes[i].hg[t] = tr == 0;

		}
	}
}

template<class TMes>
void FactorGraph<TMes>::drop_contacts(times_t t)
{
	for (size_t i = 0; i < nodes.size(); ++i) {
		Node & fi = nodes[i];
		for (size_t k = 0; k < fi.neighs.size(); ++k) {
			if (fi.times[fi.neighs[k].t[0]] < t)
				throw std::invalid_argument("can only drop first contact");
			else if (fi.times[fi.neighs[k].t[0]] == t) {
				fi.neighs[k].t.erase(fi.neighs[k].t.begin(), fi.neighs[k].t.begin() + 1);
				fi.neighs[k].lambdas.erase(fi.neighs[k].lambdas.begin(), fi.neighs[k].lambdas.begin() + 1);
				--fi.neighs[k].msg;
			}
		}
	}
}
#endif
