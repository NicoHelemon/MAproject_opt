**Project Overview**  
This codebase studies how to best transform the edge‐weights of a **weighted stochastic block model** (WSBM) in order to recover its underlying community labels.  It combines _theoretical_ separability measures (Chernoff information) with _empirical_ clustering performance (Rand index, GMM score) and explores a rich family of weight transformations and gating strategies.

---

## 1. Theoretical Standpoint

1. **Generative Models (WSBMs)**  
   - Two families of block models are implemented:  
     • **Beta‐WSBM** where nonzero edges are drawn from Beta distributions  
     • **Lognormal‐WSBM** where nonzero edges follow a Lognormal law  
   - Each model is parameterized by  
     – ρ: edge‐presence probability  
     – Π: community‐membership proportions  
     – distributional shape parameters (α for Beta, σ for Lognormal)  

2. **Chernoff Information as a Separability Measure**  
   - For any two communities \(k,\ell\), the _Chernoff information_  
     \(\displaystyle C_{k\ell} = \min_{t\in[0,1]}\tfrac12\,t(1-t)\,(u^\top\Pi\,S^{-1}(t)\,u)\)  
     quantifies how easily one can distinguish them.  
   - Closed‐form (or numerically optimized) formulas are provided for all transforms:
     – _True_ Chernoff information using the model’s known parameters  
     – _Empirical_ Chernoff information estimated from the adjacency matrix (graph)  
     – Chernoff information in the spectral embedding space  

3. **Family of Weight Transformations**  
   - Identity, opposite (1–A), logarithm, threshold, power‐law, ranking, and quantile thresholding  
   - Each transformation reshapes the weight distribution and thus changes the theoretical Chernoff information  

---

## 2. Experimental Pipeline

1. **Data Generation & Embedding**  
   - For each combination of \((\rho,\pi,\alpha)\) or \((\rho,\pi,\sigma)\) and random seed:  
     1. Sample a weighted SBM instance \((A,Z)\)  
     2. Apply each weight transform \(T\) to \(A\)  
     3. Compute a 2D spectral embedding of \(T(A)\)  
     4. Fit a 2‐component Gaussian Mixture Model (GMM) to the embedding → obtain \(\hat Z\) and a “GMM score”  

2. **Performance Metrics**  
   - **Rand Index** between true labels \(Z\) and \(\hat Z\)  
   - **GMM Score** (log‐likelihood)  
   - **Chernoff‐based metrics**  
     • True vs empirical vs embedding‐estimated separability  
     • “Gated” versions: multiplying Chernoff grids by a sigmoid of the GMM score to emphasize better‐clustered regions  

3. **Aggregating & Comparing Transforms**  
   - For each \((\rho,\pi)\) pair and transformation \(T\), assemble 2D grids of all metrics over a grid of model shape parameters  
   - Compute:  
     • **Best‐Transform** maps (which \(T\) maximizes each metric cell‐wise)  
     • **Regret** relative to the oracle (max over \(T\))  
     • **Average Rand & Regret statistics**  
     • **Partial Spearman correlation** and **bias** of each estimator vs the Rand index  

4. **Hyperparameter Search for Gating**  
   - The “gating” idea: weight each cell of a metric grid by  
     \(\;\sigma_w(GMM\_score - s_0)\)  
     and then choose the best transform on this weighted grid  
   - Exhaustively sweep shift \(s_0\) and width \(w\) of the sigmoid to maximize average Rand index  
   - Compare “non‐gated” vs “gated” selection, both per‐model and across models  

5. **Visualization & Analysis**  
   - Heatmaps of every metric, correlation matrices, bias maps  
   - Best‐transform heatmaps showing where each \(T\) wins  
   - Bar plots of average Rand/regret per transform  
   - Curves showing how gating hyperparameters affect Rand index (model‐wise and aggregated)  

---

## 3. What This Achieves

- **Theory → Practice Link**  
  By computing both the _true_ Chernoff information and its empirical/embedding estimates, the project tests how well these separability measures predict actual clustering success.

- **Transform Selection**  
  It systematically evaluates dozens of weight‐transform strategies, measuring their impact on community recovery across a broad parameter regime.

- **Adaptive Gating**  
  Introducing gating based on the GMM fit score allows the pipeline to “trust” certain regions of the parameter grid more and further improve transform selection.

- **Comprehensive Evaluation**  
  The combination of adjusted Rand, partial correlations, regret analysis, and visualization gives a full picture of (i) which transforms work best, (ii) how separability metrics relate to performance, and (iii) how to tune meta‐parameters (gating) to maximize clustering accuracy.

In short, this project explores **how to preprocess graph weights**—via a rich set of transforms and gating strategies—to **maximize the recoverability of hidden communities**, grounded in both **Chernoff‐information theory** and **empirical clustering experiments**.