#ifndef PARAMS_H
#define PARAMS_H

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/differentiation/autodiff.hpp>
#include <valarray>

#include <iostream>
#include <exception>
#include <memory>


typedef double real_t;
typedef int times_t;

typedef std::valarray<real_t> RealParams;

struct Proba
{
	template<class T>
	Proba(T const & n) : theta(n) {}
	virtual real_t operator()(real_t) const = 0;
	virtual RealParams const & grad(real_t d) const = 0;
	virtual void print(std::ostream &) const = 0;
	std::istream & operator>>(std::istream & ist) {
		for (int i = 0; i < int(theta.size()); ++i)
			return ist >> theta[i];
		return ist;
	}
	RealParams theta;
};

std::ostream & operator<<(std::ostream & ost, Proba const & p);

struct PriorDiscrete : public Proba
{
	PriorDiscrete(std::vector<real_t> const & p) : Proba(p.size()) { for (size_t t = 0; t < p.size(); ++t) theta[t] = p[t]; }
	PriorDiscrete(Proba const & p, int T);
	real_t operator()(real_t d) const { return d < 0 || d >= int(theta.size()) ? 0.0 : theta[d]; }
	RealParams const & grad(real_t d) const {
		int const T = theta.size();
		for (int t = 0; t < T; ++t)
			dtheta[t] = (-(T-t)*1.0/T + (t <= d));
		return dtheta;
	}
	void print(std::ostream & ost) const {
		ost << "PriorDiscrete(";
		for (size_t i = 0; i < theta.size() - 1; ++i)
			ost << theta[i] << ",";
		ost << theta[theta.size() - 1] << ")";
	}
	mutable RealParams dtheta;
};

struct Cached : public Proba
{
	Cached(std::shared_ptr<Proba> const & prob, int T) : Proba(prob->theta), prob(prob), p(T), dp(T), zero(0.0, prob->theta.size()) {
		recompute();
	}
	std::shared_ptr<Proba> prob;
	std::vector<real_t> p;
	std::vector<RealParams> dp;
	RealParams const zero;
	real_t operator()(real_t d) const { return d < 0 || d >= int(p.size()) ? 0.0: p[d]; }
	RealParams const & grad(real_t d) const { return d < 0 || d >= int(dp.size()) ? zero : dp[d]; }
	void recompute() {
		prob->theta = theta;
		for (size_t d = 0; d < p.size(); ++d) {
			p[d] = (*prob)(d);
			dp[d] = (*prob).grad(d);
		}
	}
	void print(std::ostream & ost) const { ost << "Cached(" << prob << ")"; }
};

struct Uniform : public Proba
{
	Uniform(real_t p) : Proba(RealParams({p})), dtheta(RealParams({1.0})) {}
	real_t operator()(real_t d) const { return theta[0]; }
	RealParams const & grad(real_t d) const { return dtheta; }
	void print(std::ostream & ost) const { ost << "Uniform(" << theta[0] << ")"; }
	mutable RealParams dtheta;
};




struct Exponential : public Proba
{
	Exponential(real_t mu) : Proba(RealParams({mu})), dtheta({0,1}) {}
	real_t operator()(real_t d) const { return exp(-theta[0]*d); }
	RealParams const & grad(real_t d) const { dtheta[0]= -d*exp(-theta[0]*d); return dtheta; }
	void print(std::ostream & ost) const { ost << "Exponential("<< theta[0] << ")"; }
	mutable RealParams dtheta;
};


struct Gamma : public Proba
{
	Gamma(real_t k, real_t mu) : Proba(RealParams({k,mu})), dtheta({0.0, 0.0}) {}
	real_t operator()(real_t d) const { return boost::math::gamma_q(theta[0], d * theta[1]); }
	RealParams const & grad(real_t d) const {
  		auto const x = boost::math::differentiation::make_ftuple<real_t, 1, 1>(theta[0], theta[1]);
		auto const & xk = std::get<0>(x);
		auto const & xmu = std::get<1>(x);
		auto const f = boost::math::gamma_q(xk, xmu * d);
		dtheta[0] = f.derivative(1,0);
		dtheta[1] = f.derivative(0,1);
		return dtheta;
	}
	void print(std::ostream & ost) const { ost << "Gamma(" << theta[0] << "," << theta[1] << ")"; }
	mutable RealParams dtheta;
};


struct Params {
	std::shared_ptr<Proba> prob_i;
	std::shared_ptr<Proba> prob_r;
	real_t pseed;
	real_t psus;
	real_t fp_rate;
	real_t fn_rate;
	real_t pautoinf;
	real_t learn_rate;
	Params(std::shared_ptr<Proba> const & pi, std::shared_ptr<Proba> const & pr, real_t pseed, real_t psus, real_t fp_rate, real_t fn_rate, real_t pautoinf, real_t learn_rate);
};

std::ostream & operator<<(std::ostream &, Params const &);

#endif
