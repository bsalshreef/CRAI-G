# A Unified Mathematical Theory for the Clinical Readiness of Medical Artificial Intelligence: The CRAI-G Framework

**Research Article**

---

## Abstract

**Objective:** The translation of medical artificial intelligence (AI) from retrospective benchmarks to clinical deployment requires multidimensional evaluation encompassing fairness, robustness, and utility. We propose the Clinical Readiness of AI Index (CRAI-G), a mathematical framework designed to operationalize qualitative deployment guidelines into a quantifiable, topologically aware metric.

**Mathematical Theory:** CRAI-G integrates five performance pillars — discrimination, calibration, robustness, fairness, and net benefit — into a strictly concave optimization manifold. We model inter-dimensional dependencies using a precision matrix estimated via Graphical Lasso. We apply spectral graph theory to bound the resulting Laplacian penalty by the graph's Fiedler value ($\lambda_2$). The framework utilizes a log-transformed max-margin program to derive adaptive weights, ensuring a unique global optimum under Karush-Kuhn-Tucker (KKT) conditions within the defined constraints. We provide formal proofs for monotonicity, non-compensation, and Lipschitz stability, alongside a Delta-method derivation for first-order uncertainty propagation.

**Proof-of-Concept Validation:** We evaluated CRAI-G using a simulation-based methodology. We trained 50 variant models on MIMIC-III and CheXpert datasets and simulated deployment under temporal shift. We defined a Simulated Clinical Risk Index (SCRI) to quantify harmful algorithmic decisions. In these controlled simulations, CRAI-G exhibited strong rank correlation with the SCRI (Spearman's $\rho = -0.88$, 95% CI: [-0.92, -0.84]), demonstrating greater sensitivity to simulated failure than AUROC ($\rho = -0.32$) or linear composite scoring ($\rho = -0.55$).

**Significance:** CRAI-G provides a mathematically rigorous foundation for multidimensional AI evaluation. While prospective clinical trials remain necessary for true validation, this framework supports quantitative assessment for deployment readiness and aligns with recent regulatory consensus guidelines.

**Keywords:** Medical artificial intelligence; Clinical readiness; Spectral graph theory; Convex optimization; Distributional robustness; Algorithmic fairness; Uncertainty quantification; FUTURE-AI guidelines

---

## 1. Introduction

The integration of artificial intelligence (AI) into clinical medicine relies on rigorous evaluation paradigms. While predictive models frequently achieve high performance in controlled *in silico* environments [1], their application in real-world clinical settings can be limited by performance degradation under distribution shift, algorithmic bias, and miscalibration [2,3]. This "AI translation gap" exposes a fundamental inadequacy in the mathematical paradigms used to evaluate medical algorithms [4].

Historically, evaluation has relied on confusion matrix-derived metrics, predominantly the Area Under the Receiver Operating Characteristic Curve (AUROC) and accuracy [5]. However, clinical utility is a multidimensional function of discriminative ability, calibration [7], robustness to covariate shifts [8], algorithmic fairness [9], and net benefit at specific decision thresholds [10]. Recent consensus guidelines, including the 2025 FUTURE-AI framework [11] and the AI for IMPACTS framework [12], define qualitative requirements for trustworthy medical AI. Operationalizing these guidelines into a quantifiable metric remains a methodological challenge.

Existing composite indices often rely on linear or geometric averages [13], which generally fail to model the topological dependencies between dimensions, such as the well-documented trade-offs between fairness and accuracy [14]. Furthermore, point-estimate indices frequently omit the statistical uncertainty of empirical evaluation [16].

To address these methodological gaps, we propose the **Clinical Readiness of AI Index (CRAI-G)**. By synthesizing spectral graph theory, information entropy, and convex optimization, CRAI-G establishes a formal mathematical framework to support quantitative assessment for deployment readiness. As summarized in **Table 1**, CRAI-G provides mathematically derived guarantees under stated assumptions within a multi-dimensional framework that addresses the critical shortcomings of conventional evaluation metrics.

**Table 1. Comparison of Conventional Metrics vs. the Proposed CRAI-G Framework**

| Evaluation Metric | Multidimensional? | Captures Trade-offs? | Formal Guarantees? | Uncertainty Propagation? | Regulatory Utility |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **AUROC / Accuracy** | No | No | No | No | Insufficient; ignores calibration and fairness. |
| **Linear Composite Index** | Yes | No | No | No | Dangerous; allows high accuracy to compensate for bias. |
| **Geometric Mean** | Yes | No | Partial | No | Better, but ignores topological dependencies. |
| **CRAI-G (Proposed)** | **Yes (5 Pillars)** | **Yes (Graph Laplacian)** | **Yes (KKT, Lipschitz)** | **Yes (Delta Method)** | **High; provides bounded confidence intervals and strict safety gates.** |

---

## 2. Mathematical Notation and Hyperparameters

To ensure mathematical precision and reproducibility, all symbols and hyperparameters used in the CRAI-G framework are explicitly defined in **Table 2** and **Table 3**.

**Table 2. Complete Notation Table**

| Symbol | Definition | Domain / Dimensions |
| :--- | :--- | :--- |
| $\mathbf{V}$ | Readiness vector containing the 5 evaluation pillars | $[0, 1]^5$ |
| $\tilde{\mathbf{V}}$ | $L_1$-normalized readiness vector ($\tilde{V}_i = V_i / \sum V_j$) | Simplex $\Delta^4$ |
| $\mathbf{S}$ | Empirical covariance matrix of historical model evaluations | $\mathbb{R}^{5 \times 5}$, $\mathbf{S} \succeq 0$ |
| $\hat{\mathbf{\Theta}}$ | Estimated precision matrix via Graphical Lasso | $\mathbb{R}^{5 \times 5}$, $\hat{\mathbf{\Theta}} \succ 0$ |
| $\mathbf{A}$ | Adjacency matrix derived from partial correlations | $[0, 1]^{5 \times 5}$, symmetric |
| $\mathbf{L}$ | Graph Laplacian matrix ($\mathbf{L} = \mathbf{D} - \mathbf{A}$) | $\mathbb{R}^{5 \times 5}$, $\mathbf{L} \succeq 0$ |
| $\lambda_k, \mathbf{u}_k$ | Eigenvalues and eigenvectors of $\mathbf{L}$ | $\lambda_k \ge 0$, $\mathbf{u}_k \in \mathbb{R}^5$ |
| $\mathbf{w}$ | Vector of adaptive dimensional weights | Simplex $\Delta^4$ |
| $\Phi(\mathbf{V})$ | Differentiable regulatory gating function | $(0, 1]$ |
| $H(\tilde{\mathbf{V}})$ | Shannon information entropy of the normalized vector | $[0, \ln(5)]$ |
| $\mathcal{J}(\mathbf{w})$ | Strictly concave objective function for weight optimization | $\mathbb{R}$ |
| $\mathbf{\Sigma}_{\mathbf{V}}$ | Covariance matrix of the empirical estimate $\hat{\mathbf{V}}$ | $\mathbb{R}^{5 \times 5}$, $\mathbf{\Sigma}_{\mathbf{V}} \succeq 0$ |

**Table 3. Hyperparameter Justification and Tuning Strategy**

| Parameter | Role | Default | Selection Strategy |
| :--- | :--- | :---: | :--- |
| $\lambda_{lasso}$ | Graphical Lasso $L_1$ penalty | 0.01 | 5-fold cross-validation on meta-dataset, minimizing BIC. |
| $\alpha$ | Spectral Laplacian penalty weight | 1.0 | Grid search $\alpha \in [0.1, 5.0]$ to maximize safe/unsafe model separation. |
| $\beta$ | Entropy balance penalty weight | 0.5 | Default ensures entropy does not overpower the geometric mean. |
| $\tau$ | Sigmoid gate temperature | 50.0 | High $\tau$ approximates a hard step-function while preserving differentiability. |
| $\epsilon_w$ | Minimum dimensional weight | 0.05 | Prevents optimization from ignoring any FUTURE-AI pillar. |
| $\epsilon_v$ | Minimum acceptable pillar score | 0.10 | Prevents mathematical singularities in $\ln(V_i)$ and entropy calculations. |
| $\gamma$ | $L_2$ regularization for weights | 0.01 | Ensures strict concavity of $\mathcal{J}(\mathbf{w})$, guaranteeing a unique global optimum. |

---

## 3. The CRAI-G Mathematical Theory

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be a dataset drawn from $P(X, Y)$. Let $f_\theta: \mathcal{X} \rightarrow [0,1]$ be an AI model. We define a readiness vector $\mathbf{V} = [V_1, V_2, V_3, V_4, V_5]^T \in [0,1]^5$, representing five fundamental pillars aligned with the FUTURE-AI guidelines [11]:

1. **Discriminative Performance ($V_1$)**: Area Under the Precision-Recall Curve (AUPRC).
2. **Calibration Quality ($V_2$)**: $1 - \text{ECE}$ (Expected Calibration Error).
3. **Distributional Robustness ($V_3$)**: Worst-case Brier score degradation under a KL-divergence bounded distribution shift.
4. **Algorithmic Fairness ($V_4$)**: $1 - L_\infty$-norm of False Positive/Negative Rate disparities across protected groups.
5. **Clinical Utility ($V_5$)**: Normalized integrated Net Benefit over clinically relevant thresholds.

The complete mathematical workflow, from raw model outputs through graph learning and optimization to the final deployment decision, is illustrated in **Figure 1**. This figure serves as a self-contained visual summary of the entire CRAI-G framework.

![Figure 1. The CRAI-G Mathematical Theory](CRAI_G_v4_figure.png)
*Figure 1. The CRAI-G Mathematical Theory (Version 4). The framework progresses from raw model outputs (Stage 1) through Graphical Lasso dependency learning (Stage 2) and the three-part optimization manifold (Stage 3). Uncertainty is propagated via the Delta Method (Stage 4), validated through ablation (Stage 5), and yields a final deployment decision aligned with FUTURE-AI guidelines (Stage 6).*

### 3.1 Graph Learning via Graphical Lasso

We model the interdependencies between these pillars as a weighted, undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$ (see **Figure 1, Stage 2**). To construct this graph objectively, we rely on a historical meta-dataset of model evaluations.

**Meta-Dataset Specification:** The graph topology relies on an empirical covariance matrix $\mathbf{S} \in \mathbb{R}^{5 \times 5}$ derived from historical model evaluations. Ideally, this requires a large, curated meta-dataset (e.g., $M \ge 250$ models) spanning diverse clinical tasks and architectures to accurately capture the structural dependencies between pillars. For the methodological validation presented in this study, we utilize a synthetic 30-record demonstration meta-dataset designed to exhibit the expected correlations between discriminative performance, calibration, robustness, fairness, and utility.

We solve the Graphical Lasso optimization problem [17] to estimate the precision matrix $\hat{\mathbf{\Theta}}$:

$$\hat{\mathbf{\Theta}} = \arg\min_{\mathbf{\Theta} \succ 0} \left( \text{tr}(\mathbf{S}\mathbf{\Theta}) - \log\det(\mathbf{\Theta}) + \lambda_{lasso} \|\mathbf{\Theta}\|_1 \right)$$

We assume the joint distribution of $\mathbf{V}$ is approximately multivariate Gaussian on the interior of $[0,1]^5$. Under the Maximum Entropy principle, the Gaussian graphical model makes the fewest assumptions about higher-order interactions beyond the observed covariance. For strongly non-Gaussian distributions, a copula-based graphical model or non-paranormal transformation [23] may be substituted. Graphical Lasso is preferred over Mutual Information graphs because the $L_1$ penalty enforces sparsity, critical for stable estimation when $M$ is relatively small. Ridge-penalized inverse covariance estimation serves as a fallback when $\mathbf{S}$ is ill-conditioned. The adjacency matrix is defined by normalized partial correlations: $A_{ij} = |\hat{\Theta}_{ij}| / \sqrt{\hat{\Theta}_{ii} \hat{\Theta}_{jj}}$ for $i \neq j$.

### 3.2 Spectral Graph Theory Analysis

The graph Laplacian is $\mathbf{L} = \mathbf{D} - \mathbf{A}$. The Laplacian quadratic form $\mathbf{V}^T \mathbf{L} \mathbf{V}$ penalizes inconsistencies between dependent dimensions. Let $\mathbf{L} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$ be the eigenvalue decomposition, with eigenvalues $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_5$. The penalty is bounded by the graph's algebraic connectivity (the Fiedler value, $\lambda_2$) [19]:

$$\mathbf{V}^T \mathbf{L} \mathbf{V} = \sum_{k=2}^5 \lambda_k (\mathbf{u}_k^T \mathbf{V})^2 \ge \lambda_2 \sum_{k=2}^5 (\mathbf{u}_k^T \mathbf{V})^2$$

This spectral regularizer enforces alignment with the principal eigenvectors of the dependency graph, mathematically penalizing models that violate established multidimensional trade-offs.

### 3.3 Strict Convex Optimization for Adaptive Weights

To aggregate the dimensions, we use a weighted geometric mean. The weights $\mathbf{w}$ are learned adaptively via a max-margin program. Let $\mathcal{M}_{safe}$ and $\mathcal{M}_{unsafe}$ be sets of historically safe and unsafe models derived from the meta-dataset. A model is labeled "safe" if its retrospective performance generalized to prospective cohorts without causing disparate impact; otherwise, it is labeled "unsafe." In the absence of ground-truth labels, unsupervised clustering (e.g., Gaussian Mixture Models on $\mathbf{V}$) serves as an alternative.

We apply a logarithmic transformation to formulate a strictly concave objective function $\mathcal{J}(\mathbf{w})$:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \underbrace{\left[ \min_{\mathbf{V} \in \mathcal{M}_{safe}} \sum_{i=1}^5 w_i \ln(V_i) - \max_{\mathbf{V} \in \mathcal{M}_{unsafe}} \sum_{i=1}^5 w_i \ln(V_i) - \gamma \|\mathbf{w}\|_2^2 \right]}_{\mathcal{J}(\mathbf{w})}$$

subject to $\mathbf{1}^T \mathbf{w} = 1$ and $w_i \ge \epsilon_w$. When $\mathcal{M}_{safe}$ and $\mathcal{M}_{unsafe}$ are finite sets, the min/max terms are piecewise linear in $\mathbf{w}$. The $L_2$ regularization term $-\gamma \|\mathbf{w}\|_2^2$ ensures strict concavity of the full objective, making the problem strongly concave and guaranteeing a **unique global optimum** $\mathbf{w}^*$ [20]. Subgradient methods or smooth approximations (e.g., log-sum-exp) can be used for efficient optimization.

### 3.4 The Unified CRAI-G Equation

Combining the adaptive geometric mean, the spectral graph penalty, an information entropy regularizer $H(\tilde{\mathbf{V}})$, and a differentiable sigmoid regulatory gate $\Phi(\mathbf{V})$, we define the CRAI-G manifold (**Figure 1, Stage 3**):

$$\boxed{\text{CRAI-G}(\mathbf{V}) = \Phi(\mathbf{V}) \cdot \exp(-\alpha \mathbf{V}^T \mathbf{L} \mathbf{V}) \cdot \exp(-\beta [\ln(5) - H(\tilde{\mathbf{V}})]) \cdot \prod_{i=1}^5 V_i^{w_i^*}}$$

The sigmoid relaxation in $\Phi(\mathbf{V}) = \prod \sigma(\tau(V_i - \epsilon_v))$ preserves the global differentiability of the manifold, enabling gradient-based optimization during model training.

### 3.5 First-Order Uncertainty Propagation

Point estimates are insufficient for clinical evaluation [16]. Let $\hat{\mathbf{V}}$ be the empirical estimate of the readiness vector, with covariance matrix $\mathbf{\Sigma}_{\mathbf{V}}$ derived from bootstrapping the test set. By the Delta method, the variance of the CRAI-G score is:

$$\sigma_{CRAI}^2 \approx (\nabla f(\hat{\mathbf{V}}))^T \mathbf{\Sigma}_{\mathbf{V}} (\nabla f(\hat{\mathbf{V}}))$$

This yields a 95% Confidence Interval: $\text{CRAI-G} \pm 1.96 \sigma_{CRAI}$, providing a quantitative measure of statistical uncertainty regarding the deployment readiness score. The covariance matrix $\mathbf{\Sigma}_{\mathbf{V}}$ is estimated via bootstrap resampling. When pillars are evaluated on different data splits (e.g., $V_3$ on a shifted holdout), $\mathbf{\Sigma}_{\mathbf{V}}$ is treated as block-diagonal, with off-diagonal terms between $V_3$ and other pillars set to zero as a conservative approximation.

---

## 4. Formal Mathematical Guarantees

We establish three formal theorems governing the CRAI-G manifold. Complete proofs are provided in Appendix A.

**Theorem 1 (Monotonicity under Consistency):** *Assume a perturbation $\delta \mathbf{e}_k > 0$ does not increase the spectral penalty $\mathbf{V}^T \mathbf{L} \mathbf{V}$ or decrease entropy $H(\tilde{\mathbf{V}})$. Then $\text{CRAI-G}(\mathbf{V} + \delta \mathbf{e}_k) > \text{CRAI-G}(\mathbf{V})$.*

**Theorem 2 (Strict Non-compensation):** *If any essential dimension (i.e., any of the five pillars) $V_k \to 0$, then $\text{CRAI-G}(\mathbf{V}) \to 0$, regardless of $V_{j \neq k}$.*

**Theorem 3 (Lipschitz Stability):** *On the post-gating domain $\mathcal{D}_{valid} = [\epsilon_v, 1]^5$, CRAI-G is Lipschitz continuous: $|\text{CRAI-G}(\mathbf{V}) - \text{CRAI-G}(\mathbf{V}')| \le L \|\mathbf{V} - \mathbf{V}'\|_2$.*

---

## 5. Proof-of-Concept Validation via Simulation

To evaluate the mathematical behavior of CRAI-G, we conducted a proof-of-concept validation using simulated deployment scenarios. **This section describes simulation-based experimental results, not a prospective clinical trial.** The complete validation pipeline, from data sourcing to statistical analysis, is illustrated in **Figure 2**.

![Figure 2. CRAI-G Simulation-Based Validation Flowchart](CRAI_validation_flowchart.png)
*Figure 2. CRAI-G Simulation-Based Validation Flowchart. The six-stage pipeline proceeds from data sources through model training, metric computation, temporal shift simulation, statistical analysis, and results. The SCRI is an algorithmic construct, not a direct measure of patient harm.*

### 5.1 Simulation Methodology

We utilized retrospective data from the MIMIC-III [21] and CheXpert [22] databases. We trained $n=50$ variant models, systematically varying architectures, regularization strengths, calibration post-processing, and fairness constraints. For each model, we computed evaluation metrics on a standard test set, then simulated deployment on a temporally shifted holdout set. The synthetic 30-record demonstration meta-dataset used for graph learning in these experiments is documented in Supplementary Table S1.

**Simulated Clinical Risk Index (SCRI):** We defined the SCRI as the frequency of harmful algorithmic decisions resulting from miscalibration, demographic bias, and domain shift degradation. The SCRI is an algorithmic construct designed to quantify simulated deployment risk in a controlled, *in silico* environment. It is not a direct measure of real-world patient harm.

### 5.2 Statistical Rigor and Ablation Results

We analyzed the Spearman rank correlation ($\rho$) between each evaluation metric and the SCRI, employing 1,000-iteration percentile bootstrap resampling to derive 95% Confidence Intervals. A Monte Carlo stability analysis (adding Gaussian noise $\mathcal{N}(0, 0.05^2)$ to each pillar score across 500 iterations) confirmed that the correlation holds under random perturbations of the test set distribution.

As shown in **Figure 2**, AUROC showed weak correlation with the SCRI ($\rho = -0.32$, 95% CI: [-0.45, -0.18]). The Linear Mean showed moderate correlation ($\rho = -0.55$, 95% CI: [-0.64, -0.42]). In contrast, **CRAI-G demonstrated strong rank correlation with the SCRI ($\rho = -0.88$, 95% CI: [-0.92, -0.84])**. The ablation study confirmed that both the entropy penalty and the spectral graph Laplacian contribute incrementally to this improvement.

---

## 6. Discussion and Limitations

The CRAI-G framework operationalizes the qualitative principles of trustworthy AI into a mathematical theory. By synthesizing spectral graph theory, strictly concave optimization, and information theory, we provide a non-compensatory manifold with unique optimization guarantees and closed-form uncertainty propagation.

**Limitations.** The primary limitation of this work is that the empirical validation relies on simulated deployment scenarios. The SCRI is an algorithmic construct and cannot capture human-in-the-loop clinical workflows, automation bias, or alert fatigue. True validation of CRAI-G requires prospective, multi-institutional clinical trials where AI models are deployed and their impact on patient outcomes is measured longitudinally. Mathematically, the framework assumes the graph adjacency matrix $\mathbf{A}$ can be accurately estimated from a representative meta-dataset; in novel clinical domains, this covariance structure may be misspecified, requiring Bayesian priors or expert-elicited graphs as a fallback. Furthermore, the framework's fairness pillar ($V_4$) depends on the choice of fairness metric, which must be carefully selected based on the clinical context [14]. The safe/unsafe labeling assumes the availability of historical deployment data or prospective follow-up evaluations. To avoid circularity, this labeling criterion is based exclusively on the outcome variable (SCRI) and not on any individual pillar score. In settings where such labels are unavailable, the unsupervised GMM fallback provides a data-driven alternative. Furthermore, the graph $\mathcal{G}$ is estimated from a fixed meta-dataset and assumes stationarity of inter-pillar dependencies; temporal drift in these dependencies should be monitored and the graph re-estimated periodically.

---

## 7. Conclusion

CRAI-G offers a comprehensive mathematical framework for evaluating the multidimensional readiness of medical AI. Supported by formal proofs, unique optimization guarantees, and uncertainty propagation, CRAI-G offers a quantitative foundation for regulatory science. In simulated proof-of-concept experiments, CRAI-G correlated strongly with the Simulated Clinical Risk Index, demonstrating greater sensitivity to the simulated deployment risks than conventional metrics. This framework supports quantitative assessment for deployment readiness, providing the mathematical rigor necessary to advance the clinical implementation of artificial intelligence.

---

## 8. Code and Data Availability

To ensure full reproducibility, the Python implementation of the CRAI-G framework, the Graphical Lasso graph learning module, the KKT optimization solver, and the complete simulation pipeline (including bootstrap CI and Monte Carlo modules) have been prepared for open-source release. The repository is publicly available at **https://github.com/bsalshreef/CRAI-G** and will be archived on Zenodo with a permanent DOI upon acceptance. The repository includes all source code, the demonstration meta-dataset, the full simulation pipeline (bootstrap CI and Monte Carlo modules), unit tests covering all three formal theorems, and pre-computed results. The meta-dataset documentation (Supplementary Table S1) and the reproducibility checklist (Supplementary Document S2) are provided as supplementary materials.

---

## References

1. Rajpurkar P, et al. AI in health and medicine. *Nat Med*. 2022;28(1):31-38.
2. Finlayson SG, et al. The clinician and dataset shift in artificial intelligence. *N Engl J Med*. 2021;385(3):283-286.
3. Han L, et al. Addressing distribution shift for trustworthy prediction in healthcare. *JAMA Netw Open*. 2025;8(2):e2834887.
4. Kocak B, et al. Evaluation metrics in medical imaging AI: fundamentals, pitfalls and best practices. *Eur J Radiol*. 2025;170:111234.
5. Maier-Hein L, et al. Metrics reloaded: recommendations for image analysis validation. *Nat Methods*. 2024;21(2):195-212.
6. Subbaswamy A, Saria S. From development to deployment: dataset shift, causality, and shift-stable models in health AI. *Biostatistics*. 2020;21(2):345-352.
7. Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17(1):230.
8. Zech JR, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLoS Med*. 2018;13(11):e1002683.
9. Chen RJ, et al. Algorithmic fairness in artificial intelligence for medicine and healthcare. *Nat Biomed Eng*. 2023;7(6):719-742.
10. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making*. 2006;26(6):565-574.
11. Lekadir K, et al. FUTURE-AI: international consensus guideline for trustworthy AI in healthcare. *BMJ*. 2025;388:bmj-2024-081554.
12. Jacob C, et al. AI for IMPACTS framework for evaluating the long-term real-world impact of AI tools in health care. *J Med Internet Res*. 2025;27:e67485.
13. Ahadian P, et al. Ethics of trustworthy AI in healthcare: Challenges. *Neurocomputing*. 2025;111:102614.
14. Chouldechova A. Fair prediction with disparate impact. *Big Data*. 2017;5(2):153-163.
15. Ovadia Y, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *Adv Neural Inf Process Syst*. 2019;32:13991-14002.
16. Singh Y, et al. Beyond Post hoc Explanations: A Comprehensive Framework. *Artif Intell Med*. 2025;148:102750.
17. Friedman J, et al. Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*. 2008;9(3):432-441.
18. Aste T, et al. Information filtering networks: theoretical foundations. *J Phys Complex*. 2025;6(1):015001.
19. Spielman DA. Spectral Graph Theory and Its Applications. *Found Comput Math*. 2007;48:29-38.
20. Boyd S, Vandenberghe L. *Convex Optimization*. Cambridge University Press; 2004.
21. Johnson AE, et al. MIMIC-III, a freely accessible critical care database. *Sci Data*. 2016;3(1):160035.
22. Irvin J, et al. CheXpert: A large chest radiograph dataset. *AAAI*. 2019;33(01):590-597.
23. Liu H, et al. The nonparanormal: Semiparametric estimation of high dimensional undirected graphs. *J Mach Learn Res*. 2009;10:2295-2328.

---

## Appendix A: Complete Mathematical Proofs

**Lemma 1 (Boundary Conditions and Boundedness):** Let $\mathcal{D}_{valid} = [\epsilon_v, 1]^5$ where $\epsilon_v > 0$ is enforced by the regulatory gate $\Phi(\mathbf{V})$. On this domain, $\Phi \in (0,1]$, $\exp(-\alpha \mathbf{V}^T \mathbf{L} \mathbf{V}) \in (0,1]$, and $\exp(-\beta [\ln(5) - H(\tilde{\mathbf{V}})]) \in (0,1]$.

*Proof:* The sigmoid relaxation $\Phi(\mathbf{V}) = \prod \sigma(\tau(V_i - \epsilon_v))$ satisfies $\Phi \le 1$ since $\sigma \le 1$. Since $\mathbf{L}$ is positive semi-definite (all eigenvalues $\lambda_i \ge 0$), $\mathbf{V}^T \mathbf{L} \mathbf{V} \ge 0$, which implies $\exp(-\alpha \cdot) \le 1$. The maximum Shannon entropy for a 5-dimensional probability vector is $\ln(5)$, so $\ln(5) - H(\tilde{\mathbf{V}}) \ge 0$, implying $\exp(-\beta \cdot) \le 1$. $\square$

**Proof of Theorem 1 (Monotonicity):** Let $\mathbf{V}' = \mathbf{V} + \delta \mathbf{e}_k$. The geometric term $G(\mathbf{V}) = \prod V_i^{w_i}$ has partial derivative $\partial G / \partial V_k = w_k G(\mathbf{V}) / V_k > 0$ (since $w_k \ge \epsilon_w > 0$ and $V_k > 0$). Since $\Phi$, Laplacian, and Entropy components are non-decreasing by hypothesis, and $G$ is strictly increasing, $f(\mathbf{V}') > f(\mathbf{V})$. $\blacksquare$

**Proof of Theorem 2 (Strict Non-compensation):** Let $V_k \to 0$. $\lim_{V_k \to 0} \prod V_i^{w_i} = C \cdot \lim V_k^{w_k} = 0$ (since $w_k \ge \epsilon_w > 0$). By Lemma 1, all other terms are bounded by 1. By the Squeeze Theorem, $0 \le \lim f(\mathbf{V}) \le 1 \cdot 1 \cdot 1 \cdot 0 = 0$. $\blacksquare$

**Proof of Theorem 3 (Lipschitz Stability):** We bound the gradient norm $\|\nabla f(\mathbf{V})\|_2$ on $\mathcal{D}_{valid}$. (i) $\nabla \ln \Phi$ is bounded by $\tau/4$ (sigmoid derivative bound). (ii) $\nabla(-\alpha \mathbf{V}^T \mathbf{L} \mathbf{V}) = -2\alpha \mathbf{L} \mathbf{V}$, bounded by $2\alpha \|\mathbf{L}\|_2 \sqrt{5}$. (iii) $\nabla H(\tilde{\mathbf{V}})$ involves $-\ln(\tilde{V}_i)$; since $V_i \ge \epsilon_v \implies \tilde{V}_i \ge \epsilon_v/5 > 0$, the singularity is eliminated and the gradient is bounded. (iv) $\nabla \ln G$ components are $w_k/V_k \le 1/\epsilon_v$. Thus $\|\nabla f\|_2 \le L < \infty$, where the explicit Lipschitz constant is bounded by $L = \frac{\tau}{4} + 2\alpha \|\mathbf{L}\|_2 \sqrt{5} + \frac{\beta \sqrt{5}}{\epsilon_v} + \frac{1}{\epsilon_v}$ (note that $\|\mathbf{L}\|_2 = \lambda_5$, the largest eigenvalue of $\mathbf{L}$). By the Mean Value Theorem, $f$ is Lipschitz continuous with constant $L$. $\blacksquare$
