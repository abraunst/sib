# Epidemic tracing through Statistical Inference


## Table of contents
* [Message-Passing strategies for epidemic tracing](#message-passing-strategies-for-epidemic-tracing)
* [Epidemic Control](#epidemic-control)
* [Inference of epidemic parameters](#inference-of-epidemic-parameters)
* [References](references)

## Message-Passing strategies for epidemic tracing

We show the effectiveness of strategies built on message-passing, **Belief Propagation** \[1-7\] and [**Mean-Field**](https://github.com/sphinxteam/sir_inference), and compare with a simple heuristic strategy for estimating infection risk consisting in ranking each individual by the number of contacts with other individuals that either showed symptoms of various degrees or have been tested positive:

* **BP**: we build a probability distribution over all possible histories of disease spreading \[1\]. The inference procedure consists in passing a set of so-called *cavity messages* along the edges of the network. At convergence, BP equations yield an approximation to the posterior distribution given the observations and the current estimate of transmission and recovery parameters. This method, which represents an exact Bayesian inference on networks without loops \[1,2\], has been shown to produce excellent results in a variety of partially observable settings on disease spreading;
* **MF**: Generalization of time-forward evolution equations for factorized (mean-field) probability marginals plus a backward propagation of the effects of evidences due to observations. See https://github.com/sphinxteam/sir_inference for more details;
* **Tracing**: at time t, individuals are ranked according to the number of contacts with individuals who tested positive in the time interval \[t−τ, t\].

## ROCS
ROC curves and precisions at differents epidemic sizes:

<p align="center">
  <img src="./figs/roc_.png" width="800" height="200">
</p>

[comment]: # "Averaged ROC area at different epidemic size, changing app adoptions (100\%,66\%, 62\%, 55\%)"

[comment]: # "![](./figs/auc.gif)"

## Epidemic control

We employ realistic individual-based models to investigate a number of intervention strategies aiming at containing epidemic outbreaks, such as case-based measures (e.g. individual and household quarantine and mobility restrictions).  

[OpenABM-Covid19](https://github.com/BDI-pathogens/OpenABM-Covid19) is an agent-based model (ABM) developed by Oxford's [Fraser group](https://www.coronavirus-fraser-group.org/) to simulate the spread of Covid-19 in a urban population.

An Intervention API that allows testing of risk assessment and quarantine strategies can be found at the following link: https://github.com/aleingrosso/OpenABM-Covid19.  

Intervention results in a network of 50K nodes:

<p align="center">
  <img src="figs/anim_50K_log.gif">
</p>

The population is seeded with 10 infected individuals at day 0. Intervention strategies start at day 10.

![intervention_multiple_50K](figs/N50K_o400_linear_and_log.svg)


## Inference of epidemic parameters

From the BP messages it is possible to compute an approximation of the log-likelihood of the data, the so-called *Bethe free entropy*. This quantity allows us to infer the hyper-parameters of the epidemic model such as the infection rate or the recovery rate thorugh a gradient ascent of the log-likelihood with respect to the target parameters. 

As an example of the effectiness of this procedure, we report here a plot of the Bethe free entropy as a function of the shape and rate parameters (𝛼, 𝛽) of the Gamma distribution of the recovery time. The blue point in the 2D plot corresponds to the maximum log-likelihood point.

The temporal graph used here consists in 10K nodes and all contacts up to day T = 30. After 5 initial days, 10 % of the nodes are selected uniformly at random and observed on a daily basis. 
We also report the values of the AUC in the (𝛼, 𝛽) plane associated with the inference of the infected nodes at time T = 30.


<p align="center">
  <img src="figs/inference_parameters_openABM_gamma.png" width="700" height="550">
</p>


[comment]: # "![inference_auc_parameters_10K](figs/inference_parameters_openABM_gamma.png)"



## References

\[1\] [Epidemic mitigation by statistical inference from contact tracing data, A Baker, I Biazzo, A Braunstein, G Catania, L Dall'Asta, A Ingrosso, F Krzakala, F Mazza, M Mézard, AP Muntoni, M Refinetti, S Sarao Mannelli, L Zdeborová, preprint ArXiv 2009.09422](https://arxiv.org/abs/2009.09422)

\[2\] [Inference of causality in epidemics on temporal contact networks, A Braunstein, A Ingrosso, Scientific reports 6, 27538, ](https://www.nature.com/articles/srep27538)

\[3\] [Bayesian inference of epidemics on networks via belief propagation, F Altarelli, A Braunstein, L Dall’Asta, A Lage-Castellanos, R Zecchina. Physical review letters 112 (11), 118701](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.112.118701)

\[4\] [The patient-zero problem with noisy observations. F Altarelli, A Braunstein, L Dall’Asta, A Ingrosso, R Zecchina. Journal of Statistical Mechanics: Theory and Experiment 2014 (10), P10016](https://iopscience.iop.org/article/10.1088/1742-5468/2014/10/P10016/meta)

\[5\] [Containing epidemic outbreaks by message-passing techniques. F Altarelli, A Braunstein, L Dall’Asta, JR Wakeling, R Zecchina. Physical Review X 4 (2), 021024](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.4.021024)

\[6\] [Predicting epidemic evolution on contact networks from partial observations. J Bindi, A Braunstein, L Dall’Asta. Plos one 12 (4), e0176376](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0176376)

\[7\] [Large deviations of cascade processes on graphs. F Altarelli, A Braunstein, L Dall’Asta, R Zecchina. Physical Review E 87 (6), 062115](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.87.062115)

\[8\] [Optimizing spread dynamics on graphs by message passing. F Altarelli, A Braunstein, L Dall’Asta, R Zecchina. Journal of Statistical Mechanics: Theory and Experiment 2013 (09), P09011](https://iopscience.iop.org/article/10.1088/1742-5468/2013/09/P09011/meta)


