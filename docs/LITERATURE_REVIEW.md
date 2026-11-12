# Literature review (verified sources)

Every source below was checked against a publisher, DOI or repository record on 2026-11-07.
Preprints and non-peer-reviewed work are labelled as such. Each entry says which design choice in this
project it supports.

## 1. Formula One performance: car vs driver, and the role of the grid

| Source | What it shows | How it shaped this project |
|---|---|---|
| Bell, A., Smith, J., Sabel, C. E. & Jones, K. (2016). *Formula for success: Multilevel modelling of Formula One driver and constructor performance, 1950–2014.* Journal of Quantitative Analysis in Sports 12(2), 99–112. doi:10.1515/jqas-2015-0050 | A cross-classified multilevel model over 1950–2014 finds that **team effects outweigh driver effects**, increasingly over time; the team advantage may shrink in wet races and on street circuits. | Motivates the **constructor feature family** (team form, standings, previous-season rank) next to driver features, and pre-start **weather**. Our ablation (`outputs/tables/ablation_groups.csv`) tests whether constructor features add value beyond the grid. |
| van Kesteren, E.-J. & Bergkamp, T. (2023). *Bayesian analysis of Formula One race results: disentangling driver skill and constructor advantage.* Journal of Quantitative Analysis in Sports 19(4), 273–293. doi:10.1515/jqas-2022-0021 (preprint arXiv:2203.08489) | A Bayesian multilevel rank-ordered model of 2014–2021 results attributes most of the variance in results (≈88% in the published abstract) to the **constructor**. | Same conclusion in the modern era; supports team-level features and explains why driver identity alone is weak. It is also an example of modelling the **whole finishing order** (rank-ordered logit), which we approximate by race-level normalisation (§4). |
| Henderson, D. A. & Kirrane, L. J. (2018). *A comparison of truncated and time-weighted Plackett–Luce models for probabilistic forecasting of Formula One results.* Bayesian Analysis 13(2), 335–358. doi:10.1214/17-BA1048 | Plackett–Luce models forecast whole F1 finishing orders (2010–2013 seasons); **time-weighting recent races** and truncation change forecast quality. | Supports (a) **recency-weighted form features** (last 3/5/10 races) rather than career-only averages, and (b) treating winner probabilities as a distribution over the field. Our win normalisation `p_i^τ / Σ p_j^τ` is the first-place probability of a Plackett–Luce model with strengths `p_i^τ`. |
| Weissbock, J. (2025). Submission to the Wharton Sports Analytics conference (not peer-reviewed; https://wsb.wharton.upenn.edu/wp-content/uploads/2025/12/WEISSBOCK_WSA_F1-Submission.pdf) | Re-examines how strongly starting grid determines results; argues qualifying performance may matter more than the grid slot itself. | Reason to include **qualifying gap to pole (%)** as well as grid position, and to test grid-only baselines. Used as context only. |

## 2. Data leakage and evaluation design

| Source | What it shows | How it shaped this project |
|---|---|---|
| Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). *Leakage in data mining: formulation, detection, and avoidance.* ACM TKDD 6(4), Article 15. doi:10.1145/2382577.2382579 | Defines leakage as information about the target that would not be legitimately available at prediction time; recommends a strict **learn–predict separation** and detection tests. | The T0 prediction point, the "aggregate → shift → roll" feature rule, and the **automated leakage tests T1–T6** (`tests/test_leakage.py`). At inference, inputs and outcomes are stored separately, and a test shows that scrambling every outcome column leaves predictions unchanged (`tests/test_api.py`). |
| Roberts, D. R. et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40(8), 913–929. doi:10.1111/ecog.02881 | Random CV underestimates error when data have dependence structure; recommends **blocked CV** that respects the structure. | Folds are **whole seasons** in an expanding window (train < S, validate S), never random rows. All rows of a race stay together, and confidence intervals **bootstrap races, not rows**. |

## 3. Algorithms compared

| Source | Relevance |
|---|---|
| Breiman, L. (2001). *Random Forests.* Machine Learning 45(1), 5–32. doi:10.1023/A:1010933404324 | Bagging plus random feature subsets lowers the variance of deep trees. This is the Random Forest in our ladder, and it is compared against a single decision tree to show the variance reduction. |
| Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine.* Annals of Statistics 29(5), 1189–1232. doi:10.1214/aos/1013203451 | Gradient boosting fits trees sequentially to the gradient of the loss. Our "gradient boosting" is scikit-learn's histogram-based implementation of this algorithm. |

## 4. Probability quality and calibration

| Source | What it shows | How it shaped this project |
|---|---|---|
| Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting good probabilities with supervised learning.* Proc. ICML 2005, 625–632. doi:10.1145/1102351.1102430 | Raw scores from boosted trees, random forests and other learners are often **mis-calibrated**; Platt scaling or isotonic regression fixes this, and after calibration boosted trees and random forests give some of the best probabilities. | We do not show raw classifier scores as probabilities. Win scores are **race-normalised with a temperature τ fitted on out-of-fold seasons** (a one-parameter, Platt-like calibration at race level). Podium scores are rescaled to exactly three places per race. Reliability diagrams and Brier scores are reported (`outputs/figures/models/calibration.png`). |

## 5. Weather data

| Source | Relevance |
|---|---|
| Hersbach, H. et al. (2020). *The ERA5 global reanalysis.* Quarterly Journal of the Royal Meteorological Society 146, 1999–2049. doi:10.1002/qj.3803 | Open-Meteo's historical archive is ERA5-based reanalysis (about 31 km native resolution): a reconstruction, **not a forecast**. So only the **pre-start window** (start −3 h to −1 h) is used as a model input, and race-window weather appears only in the oracle ablation E6. |

## 6. How the literature maps onto our findings and limitations

- **Feature selection:** constructor strength is the dominant driver of results (Bell et al., van Kesteren & Bergkamp). Our ablation and permutation importance show whether constructor and championship features add information beyond grid and qualifying.
- **Model comparison:** we do not assume complex beats simple. Paired race-bootstrap CIs (`model_paired_bootstrap.csv`) decide whether a difference is real (Roberts et al. on dependent data).
- **Leakage prevention:** Kaufman et al. → T0 definition, shift-before-roll, test lock on 2023–2025.
- **Chronological evaluation:** Roberts et al. → season-blocked expanding-window CV.
- **Calibration:** Niculescu-Mizil & Caruana → race-level temperature calibration and reliability diagrams.
- **Limitations:** a full Plackett–Luce or rank-ordered likelihood (Henderson & Kirrane; van Kesteren & Bergkamp) would model the joint finishing order. We use per-driver models plus race-level normalisation, which is simpler but ignores dependence between drivers. This is noted as future work.

## 7. Custom race simulation (added 2026-11-07, Mode B)

| Source | What it establishes | How it is used |
|---|---|---|
| Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis.* New York: Wiley. | The choice axiom: the relative odds of choosing one option over another do not depend on which other options are available (independence of irrelevant alternatives). | Justifies predicting an **arbitrary hypothetical field**. A ranking model fitted on real full fields remains a valid Plackett–Luce model when restricted to any subset of drivers. Verified empirically in `scenario_subset_experiment.csv` and by `tests/test_ranking.py`. |
| Plackett, R. L. (1975). *The analysis of permutations.* Journal of the Royal Statistical Society Series C (Applied Statistics) 24(2), 193–202. doi:10.2307/2346567 | Probability models for whole permutations, built by choosing the next-ranked item sequentially. | The finishing-order model in `src/ranking.py`: P(order) = Π_k exp(u_(k)) / Σ_{j≥k} exp(u_(j)), fitted by maximum likelihood (analytic gradient, L-BFGS). |
| Yellott, J. I. (1977). *The relationship between Luce's choice axiom, Thurstone's theory of comparative judgment, and the double exponential distribution.* Journal of Mathematical Psychology 15, 109–144. doi:10.1016/0022-2496(77)90026-8 | Luce's choice model is equivalent to a random-utility model with independent double-exponential (Gumbel) noise. | Exact Monte Carlo sampling of finishing orders by sorting u_i + Gumbel noise. The position distributions, podium and top-5 probabilities come from these samples. |
| Henderson, D. A. & Kirrane, L. J. (2018), see §1 | Truncated and time-weighted Plackett–Luce variants change F1 forecast quality. | Both ideas were tested: a time-weighted likelihood (half-life 8 seasons) and truncation at the top k places. Truncating at k = 1 gave the best-calibrated winner probabilities on 2017–2022 (`scenario_model_comparison.csv`). |

**Counterfactual limits.** None of these sources (and no data) can validate a race between drivers from different eras: such races never took place. The model's era-relative inputs (finish percentiles, win rates, share of the season champion's points) make the comparison defined, but not verifiable. The website states this whenever a field spans more than 10 seasons.

## 8. Separating driver from car (added 2026-11-07, D-021)

Bell et al. (2016) and van Kesteren & Bergkamp (2023) both show that most of the variance in F1 results belongs to the **constructor**, and both separate driver from team by using drivers who shared a team. The full-grid simulator applies the same idea in a simpler, transparent way. A driver is rated only against teammates in the same car, and the ratings are linked across careers by a regularised least-squares fit over all teammate pairs (`src/driver_ratings.py`). Car strength comes only from the car's season. The car's results are partly adjusted for the skill of the drivers who raced it ("crew skill"). This is what makes "any driver in any car" defined. It does **not** make it verifiable. Teammate comparisons cannot separate a driver from an unusually strong (or weak) teammate perfectly, and no data exist for drivers in cars they never raced.
