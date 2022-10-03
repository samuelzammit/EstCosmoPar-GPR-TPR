# EstCosmoPar-GPR-TPR
Estimation of Cosmological Priors using Gaussian Processes and Viable Alternatives

Several astrophysical literature is dedicated to obtaining accurate and reliable estimates for the Hubble parameter $H_0$, using diverse data sources such as CC, SNIa, and BAO, and different methodologies. This results in estimates which do not agree with each other - the so-called "$H_0$ tension". In this work, methods already established in literature for estimating $H_0$, such as Gaussian process regression (GPR) and Markov chain Monte Carlo (MCMC) methods based on the concordance $\Lambda CDM$ model, together with two novel approaches in the field, are assessed. The first novel approach makes use of non-parametric MCMC inference on the hyperparameters of a Gaussian process kernel, independently of any cosmological model such as $\Lambda CDM$. The second approach is Student's $t$-process regression (TPR), which is similar to GPR but makes use of the Student's $t$-distribution instead of the Gaussian distribution. TPR does not automatically assume Gaussianity of underlying observations and has the additional advantage of being a more generalised and flexible form of GPR. A comparison of the novel and tried-and-tested methods will be provided, as well as challenges posed by both novel approaches.
