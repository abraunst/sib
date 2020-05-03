#ifndef PARAMS_H
#define PARAMS_H

#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <exception>


typedef double real_t;
class Uniform;
class Exponential;
class Gamma;
class PriorDiscrete;


struct PriorDiscrete
{
	PriorDiscrete(std::vector<real_t> const & p) : p(p) {}
	PriorDiscrete() : p({1.0, 1.0}) {}
	real_t operator()(real_t d) const { return d < 0 || d >= int(p.size()) ? 0.0 : p[d]; }
	std::vector<real_t> p;
};

std::ostream & operator<<(std::ostream & ost, PriorDiscrete const & p);

struct Uniform
{
	Uniform(real_t p) : p(p) {}
	real_t p;
	real_t operator()(real_t d) const { return p; }
	std::istream & operator>>(std::istream & ist) { return ist >> p; }
};


std::ostream & operator<<(std::ostream & ost, Uniform const & u);


struct Exponential
{
	Exponential(real_t mu) : mu(mu) {}
	real_t mu;
	real_t operator()(real_t d) const { return exp(-mu*d); }
	std::istream & operator>>(std::istream & ist) { return ist >> mu; }
};

std::ostream & operator<<(std::ostream & ost, Exponential const & e);

struct Gamma
{
	real_t k;
	real_t mu;
	Gamma(real_t k, real_t mu) : k(k), mu(mu) {}
	real_t operator()(real_t d) const { return 1-boost::math::gamma_p(k,d*mu); }
	std::istream & operator>>(std::istream & ist) { return ist >> k >> mu; }
};

std::ostream & operator<<(std::ostream & ost, Gamma const & g);


template<class Pi, class Pr>
struct Params {
	Pi prob_i;
	Pr prob_r;
	real_t pseed;
	real_t psus;
	Params(Pi const & pi, Pr const & pr, real_t pseed, real_t psus) : prob_i(pi), prob_r(pr), pseed(pseed), psus(psus) {
		if (pseed + psus > 1)
			throw std::domain_error("pseed and psus are exclusive events but pseed+psus>1");
	}
};

template<class Pi, class Pr>
std::ostream & operator<<(std::ostream &, Params<Pi,Pr> const &);

#include "params_impl.h"

#endif
