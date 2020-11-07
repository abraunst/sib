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
#include "bp.h"
#include "cavity.h"

using namespace std;


int const Tinf = 1000000;


template<class T>
void cumsum(Message<T> & m, int a, int b)
{
	T r = m(0, 0);
	for (int sij = m.qj - 2; sij >= b; --sij)
		m(m.qj - 1, sij) += m(m.qj -1, sij + 1);
	for (int sji = m.qj - 2; sji >= a; --sji) {
		r = m(sji, m.qj - 1);
		m(sji, m.qj - 1) += m(sji + 1, m.qj - 1);
		for (int sij = m.qj - 2; sij >= b; --sij) {
			r += m(sji, sij);
			m(sji, sij) = r + m(sji + 1, sij);
		}
	}
}


BPMes & operator++(BPMes & msg)
{
	int oldqj = msg.qj;
	msg.qj++;
	int qj = msg.qj;
	msg.resize(msg.qj * msg.qj);

	//msg(sji, sij) = msg[qj * sij + sji]
	for (int sij = oldqj - 1; sij >= 0; --sij) {
		for (int sji = oldqj - 1; sji >= 0; --sji) {
			msg(sji, sij) = msg[oldqj * sij + sji];
		}
	}
        msg(qj - 1, qj - 1) = msg(qj - 2, qj - 2);
	for (int s = 0; s < qj; ++s) {
		msg(s, qj - 1) = msg(s, qj - 2);
		msg(qj - 1, s) = msg(qj - 2, s);
	}
	return msg;
}


BPMes & operator--(BPMes & msg)
{
	int qj = msg.qj;
	msg.qj--;
	for (int sij = 0; sij < qj - 1; ++sij) {
		for (int sji = 0; sji < qj - 1; ++sji) {
			msg(sji, sij) = msg[qj * (sij + 1) + (sji + 1)];
		}
	}
	msg.resize(msg.qj * msg.qj);
	return msg;
}




ostream & operator<<(ostream & o, vector<real_t> const & m)
{
	o << "{";
	for (int i=0; i<int(m.size()); ++i)
		o << m[i] << " ";
	o << "}";
	return o;
}

void update_limits(int ti, NodeType<BPMes> const &f, vector<int> & min_in, vector<int> & min_out)
{
	int n = min_in.size();
	for (int j = 0; j < n; ++j) {
		NeighType<BPMes> const & v = f.neighs[j];
		int qj = v.t.size();
		int const *b = &v.t[0];
		int const *e = &v.t[0] + qj - 1;
		min_in[j] = lower_bound(b + min_in[j], e, ti) - b;
		min_out[j] = min_in[j] + (v.t[min_in[j]] == ti && min_in[j] < qj - 1);
	}
}

template<>
real_t BPGraph::update(int i, real_t damping, bool learn)
{
	Node & f = nodes[i];
	int const n = f.neighs.size();
	int const qi = f.bt.size();

	RealParams const zero_r = RealParams(0.0, f.prob_r->theta.size());
	RealParams const zero_i = RealParams(0.0, f.prob_i->theta.size());
	// allocate buffers
	vector<BPMes> UU, HH, M, R;
	vector<Message<RealParams>> dM, dR;
	vector<real_t> ut(qi), ug(qi);
	vector<vector<real_t>> CG0, CG01;
	vector<RealParams> dC0, dC1;
	for (int j = 0; j < n; ++j) {
		Neigh const & v = nodes[f.neighs[j].index].neighs[f.neighs[j].pos];
		v.lock();
		HH.push_back(v.msg);
		v.unlock();
		UU.push_back(BPMes(v.t.size()));
		R.push_back(BPMes(v.t.size()));
		M.push_back(BPMes(v.t.size()));
		CG0.push_back(vector<real_t>(v.t.size() + 1));
		CG01.push_back(vector<real_t>(v.t.size() + 1));
		if (learn) {
			dR.push_back(Message<RealParams>(v.t.size(), zero_r));
			dM.push_back(Message<RealParams>(v.t.size(), zero_r));
			dC0.push_back(zero_i);
			dC1.push_back(zero_i);
		}
	}
	vector<real_t> C0(n), P0(n); // probas tji >= ti for each j
	vector<real_t> C1(n), P1(n); // probas tji > ti for each j
	vector<int> min_in(n), min_out(n);
	vector<real_t> ht = f.ht;

	// apply external fields
	ht[0] *= params.pseed;
	for (int t = 1; t < qi - 1; ++t)
		ht[t] *= 1 - params.pseed - params.psus;
	ht[qi-1] *= params.psus;

	// main loop
	real_t za = 0.0;
	RealParams dzr = zero_r, dp1 = zero_r, dp2 = zero_r;
	RealParams dzi = zero_i, dl = zero_i, dpi = zero_i, dlpi = zero_i;
	for (int ti = 0; ti < qi; ++ti) if (ht[ti]) {
		Proba const & prob_i = ti ? *f.prob_i : *f.prob_i0;
		Proba const & prob_r = ti ? *f.prob_r : *f.prob_r0;
		bool const dolearn = (ti > 0) && learn;
		update_limits(ti, f, min_in, min_out);

		for (int j = 0; j < n; ++j) {
			BPMes & m = M[j]; // no need to clear, just use the bottom right corner
			BPMes & r = R[j];
			Neigh const & v = f.neighs[j];
			BPMes const & h = HH[j];
			int const qj = h.qj;

			real_t pi = 1;
			dpi = zero_i;

			Message<RealParams> & dm = dM[j];
			Message<RealParams> & dr = dR[j];
			for (int sij = min_out[j]; sij < qj - 1; ++sij) {
				int tij = v.t[sij];
				real_t const l = prob_i(f.times[tij]-f.times[ti]) * v.lambdas[sij];
				for (int sji = min_in[j]; sji < qj; ++sji) {
					m(sji, sij) = l * pi * h(sji, sij);
					r(sji, sij) = l * pi * h(sji, qj - 1);
				}
				if (dolearn) {
					prob_i.grad(dl, f.times[tij]-f.times[ti]);
					dl *= v.lambdas[sij];
					dlpi = dl * pi + l * dpi;
					for (int sji = min_in[j]; sji < qj; ++sji) {
						//grad m & r
						dm(sji, sij) = dlpi * h(sji, sij);
						dr(sji, sij) = dlpi * h(sji, qj - 1);
					}
					dpi = dpi * (1 - l) - pi * dl;
				}
				pi *= 1 - l;
			}

			for (int sji = min_in[j]; sji < qj; ++sji) {
				m(sji, qj - 1) = pi * h(sji, qj - 1);
				r(sji, qj - 1) = pi * h(sji, qj - 1);
				if (dolearn) {
					dm(sji, qj - 1) = dpi * h(sji, qj - 1);
					dr(sji, qj - 1) = dpi * h(sji, qj - 1);
				}
			}

			cumsum(m, min_in[j], min_out[j]);
			cumsum(r, min_in[j], min_out[j]);
			//grad m & r
			if (dolearn) {
				cumsum(dm, min_in[j], min_out[j]);
				cumsum(dr, min_in[j], min_out[j]);
			}
			fill(CG01[j].begin(), CG01[j].end(), 0.0);
			fill(CG0[j].begin(), CG0[j].end(), 0.0);
		}
		auto min_g = min_out;
		real_t p0full = 0.0, p1full = 0.0;
		bool changed = true;
		for (int j = 0; j < n; ++j)
			--min_g[j];
		for (int gi = ti; gi < qi; ++gi) if (f.hg[gi]) {
			for (int j = 0; j < n; ++j) {
				Neigh const & v = f.neighs[j];
				int const qj = v.t.size();
				int const *b = &v.t[0];
				int newming = upper_bound(b + max(0, min_g[j]), b + qj - 1, gi) - b;
				if (newming == min_g[j])
					continue;
				min_g[j] = newming;
				changed = true;
				BPMes & m = M[j];
				BPMes & r = R[j];
				//grad m & r
				/*
				   .-----min_out
				   |   .-- min_g
				   sij     v   v
				   . . . . . . . .
				   sji. . . . . . . .
				   . . . . . . . .
				   . . . . a a b b <- min_in
				   . . . . c c d d <- min_out
				   . . . . c c d d
				   . . . . c c d d
				   . . . . c c d d


				   C0 = a + c + b' + d' = (a + c + b + d) - (b + d) + (b' + d')
				   C1 = c + d'          = c + d           - d       + d'
				   */
				C0[j] = m(min_in[j],  min_out[j]) - m(min_in[j],  min_g[j]) + r(min_in[j],  min_g[j]);
				C1[j] = m(min_out[j], min_out[j]) - m(min_out[j], min_g[j]) + r(min_out[j], min_g[j]);
				//grad C
				if (dolearn) {
					auto & dm = dM[j];
					auto & dr = dR[j];
					dC0[j] = dm(min_in[j],  min_out[j]) - dm(min_in[j],  min_g[j]) + dr(min_in[j],  min_g[j]);
					dC1[j] = dm(min_out[j], min_out[j]) - dm(min_out[j], min_g[j]) + dr(min_out[j], min_g[j]);
				}
			}
			if (changed) {
				changed = false;
				p0full = cavity(C0.begin(), C0.end(), P0.begin(), 1.0, multiplies<real_t>());
				p1full = cavity(C1.begin(), C1.end(), P1.begin(), 1.0, multiplies<real_t>());
			}
			//messages to ti, gi
			auto const d1 = f.times[gi] - f.times[ti];
			real_t const pg = gi < qi - 1 ? prob_r(d1) -  prob_r(f.times[gi + 1] - f.times[ti]) : prob_r(d1);
			real_t const c = ti == 0 || ti == qi - 1 ? p0full : (p0full - p1full * (1 - params.pautoinf));
			ug[gi] += ht[ti] * pg * c;
			ut[ti] += f.hg[gi] * pg * c;
			real_t const b = ht[ti] * f.hg[gi] * pg;
			za += b * c;
			if (dolearn) {
				//grad theta_r
				prob_r.grad(dp1, d1);
				if (gi < qi - 1) {
					auto const d2 = f.times[gi + 1] - f.times[ti];
					prob_r.grad(dp2, d2);
					dzr += ht[ti] * f.hg[gi] * (dp1 - dp2) * c;
				} else {
					dzr += ht[ti] * f.hg[gi] * dp1 * c;
				}
				//grad theta_i
				for (int j = 0; j < n; ++j) {
					dzi += b * P0[j] * dC0[j];
					if (0 < ti && ti < qi - 1)
						dzi -= b * P1[j] * dC1[j] * (1 - params.pautoinf);
				}
			}
			for (int j = 0; j < n; ++j) {
				CG0[j][min_g[j]] += b * P0[j];
				CG01[j][min_g[j]] += b * (P0[j] - P1[j] * (1 - params.pautoinf));
			}
		}
		//messages to sij, sji
		for (int j = 0; j < n; ++j) {
			partial_sum(CG0[j].rbegin(), CG0[j].rend(), CG0[j].rbegin());
			partial_sum(CG01[j].rbegin(), CG01[j].rend(), CG01[j].rbegin());
			Neigh const & v = f.neighs[j];
			int const qj = v.t.size();
			for (int sji = min_in[j]; sji < qj; ++sji) {
				// note: ti == qi - 1 implies ti == v.t[sji]
				vector<real_t> const & CG = ti == 0 || ti == v.t[sji] ? CG0[j] : CG01[j];
				real_t pi = 1;
				real_t c = 0;
				for (int sij = min_out[j]; sij < qj - 1; ++sij) {
					int const tij = v.t[sij];
					real_t const l = prob_i(f.times[tij] - f.times[ti]) * v.lambdas[sij];
					//note: CG[sij + 1] counts everything with gi >= sij
					UU[j](sij, sji) += CG[sij + 1] * pi * l;
					c += (CG[0] - CG[sij + 1]) * pi * l;
					pi *= 1 - l;
				}
				UU[j](qj - 1, sji) += c + CG[0] * pi;
			}
		}
	}
	f.f_ = log(za);
	//apply external fields on t,h
	for (int t = 0; t < qi; ++t) {
		ut[t] *= ht[t];
		ug[t] *= f.hg[t];
	}
	//update parameters
	if (learn && za) {
		f.df_r = dzr/za;
		f.df_i = dzi/za;
	}

	//compute beliefs on t,g
	real_t diff = max(setmes(ut, f.bt, damping), setmes(ug, f.bg, damping));
	f.err_ = diff;
	for (int j = 0; j < n; ++j) {
		Neigh & v = f.neighs[j];
		v.lock();
		// diff = max(diff, setmes(UU[j], v.msg, damping));
		setmes(UU[j], v.msg, damping);
		v.unlock();

		real_t zj = 0; // z_{(sij,sji)}}
		int const qj = v.t.size();
		for (int sij = 0; sij < qj; ++sij) {
			for (int sji = 0; sji < qj; ++sji) {
				zj += HH[j](sij, sji)*v.msg(sji, sij);
			}
		}
		f.f_ -= 0.5*log(zj); // half is cancelled by z_{a,(sij,sji)}
	}

	return diff;

}


template<>
real_t BPGraph::loglikelihood() const
{
	real_t L = 0;
	for(auto nit = nodes.begin(), nend = nodes.end(); nit != nend; ++nit)
		L += nit->f_;
	return L;
}


