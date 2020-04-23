
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <getopt.h>
#include <stdbool.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "bp.h"

using namespace std;

pair<vector<tuple<int,int,int,real_t> >, vector<tuple<int,int,int> > >
read_files(char const * cont_file, char const * obs_file)
{
	string line;

	ifstream cont(cont_file);
	if (!cont.is_open()) {
		cerr << "Error opening " << cont_file << endl;
		exit(EXIT_FAILURE);
	}

	auto contacts = vector<tuple<int,int,int,real_t> >();
	auto observations = vector<tuple<int,int,int> >();

	int nlines = 0;
	while (getline(cont, line)) {
		nlines++;
		if (nlines > 1) {
			stringstream s(line);
			int i, j, t;
			char g1, g2, g3;
			real_t lambda;
			s >> i >> g1 >> j >> g2 >> lambda >> g3 >> t;
			//cout << i << " " << j << " " << lambda << " " << t << endl;
			contacts.push_back(make_tuple(i,j,t,lambda));
		}
	}
	cont.close();

	ifstream obs(obs_file);
	if (!obs.is_open()) {
		cerr << "Error opening " << obs_file << endl;
		exit(EXIT_FAILURE);
	}
	nlines = 0;
	while (getline(obs,line)) {
		nlines++;
		if(nlines > 1) {
			stringstream s(line);
			int i, state, t;
			char g1, g2;
			s >> i >> g1 >> state >> g2 >> t;
			//cout << i << state << t << endl;
			observations.push_back(make_tuple(i,state,t));
		}
	}
	return make_pair(contacts,observations);
}

tuple<Params,char const *, char const *>
parse_opt(int & argc, char ** argv)
{
	Params p;
	char const * obs_file = "/dev/null";
	char const * cont_file = "/dev/null";
	int c;

	while ((c = getopt(argc, argv, "s:i:m:o:c:t:h")) != -1 ) {
		switch(c) {
			case 't':
				p.tol = stod(string(optarg));
				break;
			case 'i':
				p.maxit = stod(string(optarg));
				break;
			case 'm':
				p.mu = stod(string(optarg));
				break;
			case 's':
				p.pseed = stod(string(optarg));
				break;
			case 'o':
				obs_file = optarg;
				break;
			case 'c':
				cont_file = optarg;
				break;
			case 'h':
				cout << "SIR inference, continuous time" << endl;
				cout << "-c : Contact file with format 'i,j,lambdaij,t' " << endl;
				cout << "-o : Observation file with format 'i,state,t' " << endl;
				cout << "-m : mu parameter " << endl;
				cout << "-t : tolerance for convergence " << endl;
				cout << "-i : max iterations for convergence " << endl;
				exit(1);
			default:
				exit(1);
		}
	}
	return make_tuple(p, cont_file, obs_file);
}


int main(int argc, char ** argv) {
	auto r = parse_opt(argc, argv);
	auto co = read_files(get<1>(r), get<2>(r));
	FactorGraph factor(get<0>(co), get<1>(co), get<0>(r));
	factor.init();
	factor.iterate();
	factor.show_beliefs(cout);
	return 0;
}


