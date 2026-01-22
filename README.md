# Multilevel Monte Carlo Unbiased Gradient Estimation (MLMC) for Deep Latent Variable Models

Unbiased gradient estimation for latent variable models using Multilevel Monte Carlo (MLMC), reproducing the first numerical experiment from Shi & Cornish (2021) and comparing unbiased estimators against IWAE.

## What’s inside
- Unbiased gradient estimation with MLMC for latent variable models
- Comparison of four estimators: **ML-SS**, **ML-RR**, **SUMO**, and **IWAE**
- Bias/variance vs. compute trade-offs and gradient accuracy near the true parameter
- SGD behavior with unbiased vs. biased gradient estimates

## Contributors
Tom Rossa, Axel Pinçon and Naïl Khelifa

## References
- Shi, Yuyang, and Rob Cornish (2021), *On Multilevel Monte Carlo Unbiased Gradient Estimation for Deep Latent Variable Models*
- Rainforth, Tom, Adam R. Kosiorek, Tuan Anh Le, Chris J. Maddison, Maximilian Igl, Frank Wood, and Yee Whye Teh (2018), *Tighter Variational Bounds are Not Necessarily Better*
- Blanchet, Jose H., and Peter W. Glynn (2015), *Unbiased Monte Carlo for Optimization and Functions of Expectations via Multilevel Randomization*
- Luo, Yucen, Alex Beatson, Mohammad Norouzi, Jun Zhu, David Duvenaud, Ryan P. Adams, and Ricky T. Q. Chen (2020), *SUMO: Unbiased Estimation of Log Marginal Probability for Latent Variable Models*
- Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov (2016), *Importance Weighted Autoencoders*

MIT license, feel free to use and adapt with attribution.
