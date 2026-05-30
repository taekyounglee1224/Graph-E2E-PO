# Graph-Structured Latent Covariance Learning for GMV Portfolio Optimization

End-to-end differentiable pipeline that learns a latent covariance matrix via a Graph Neural Network and solves a Global Minimum Variance (GMV) portfolio optimization problem in a single backpropagation pass.

---

## Pipeline

```
X_t  ──┐
        ├──▶  GCNConv (2-layer)  ──▶  H_t (N×d)  ──▶  Σ_t = H_t H_t⊤ + εI  ──▶  w*
A_t  ──┘                                                (implicit, PSD by construction)
```

- **X_t** : node features (N × 7) — mean, vol, skew, kurtosis, Sharpe, momentum, max-drawdown over the lookback window  
- **A_t** : thresholded correlation adjacency (|Pearson| ≥ 0.3)  
- **H_t** : latent embeddings learned by the GNN  
- **w\*** : long-only GMV weights solved via `cvxpylayers` (differentiable QP)

Gradients flow end-to-end: $\mathcal{L} \to w^* \to H_t \to \theta_\text{GNN}$ via KKT implicit differentiation.

---

## Research Questions

| # | Question | Comparison |
|---|---|---|
| **RQ1** | Does the E2E GNN pipeline outperform classical GMV? | `Graph-L1` vs `EW` / `SampleGMV` |
| **RQ2** | Does incorporating a return signal in the loss improve risk-adjusted performance? | `Graph-L1` vs `Graph-L2`  ·  `MLP-L1` vs `MLP-L2` |
| **RQ3** | Does the graph structure provide useful inductive bias for covariance learning? | `Graph-L1` vs `MLP-L1`  ·  `Graph-L2` vs `MLP-L2` |

RQ2 and RQ3 each use **two paired comparisons** (one per architecture / loss) to verify consistency across the 2×2 factorial design.

---

## Models

Experiment design: **2×2 factorial** — {Architecture} × {Loss function}

|  | **L1 — Variance loss** | **L2 — Mean-Variance loss** |
|---|---|---|
| **Graph (GCN)** | `Graph-L1` ← proposed | `Graph-L2` ← proposed |
| **MLP** | `MLP-L1` ← RQ3 ablation | `MLP-L2` ← RQ3 ablation |

Plus two non-learning baselines: `EW` (equal-weight) and `SampleGMV` (classical GMV).

| Model | Encoder | Loss | Description |
|---|---|---|---|
| `EW` | — | — | Equal-weight (baseline) |
| `SampleGMV` | — | — | Classical GMV with sample covariance (baseline) |
| `MLP-L1` | MLP | $\mathcal{L}_1$ | E2E without graph structure — RQ3 ablation |
| `MLP-L2` | MLP | $\mathcal{L}_2$ | E2E without graph, mean-variance loss — RQ3 ablation |
| `Graph-L1` | GCN | $\mathcal{L}_1$ | E2E with graph, pure variance loss |
| `Graph-L2` | GCN | $\mathcal{L}_2$ | E2E with graph, mean-variance loss |

**Loss functions**

$$\mathcal{L}_1 = w^{*\top} \hat{\Sigma}_\text{sample} \, w^* \quad \text{(realized portfolio variance)}$$

$$\mathcal{L}_2 = w^{*\top} \hat{\Sigma}_\text{sample} \, w^* - \lambda \cdot w^{*\top} r_{t+1} \quad \text{(Markowitz mean-variance)}$$

$\hat{\Sigma}_\text{sample}$ is the sample covariance of the lookback window — fixed during the forward pass to prevent the model from gaming its own evaluation.

---

## Data

- **Source** : [Fama-French 30 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (daily returns)
- **Period** : 2000-01-03 ~ 2025-12-31 (6,539 trading days)
- **Universe** : 30 U.S. industry portfolios

```
csv/
└── 30_industry.csv   ← daily returns (%)
```

---

## Experiment Design

### Walk-Forward Expanding Window

| Fold | Train | Val | Test |
|---|---|---|---|
| 1 | 2000 ~ 2013 | 2014 ~ 2015 | 2016 ~ 2017 |
| 2 | 2000 ~ 2015 | 2016 ~ 2017 | 2018 ~ 2019 |
| 3 | 2000 ~ 2017 | 2018 ~ 2019 | 2020 ~ 2021 |
| 4 | 2000 ~ 2019 | 2020 ~ 2021 | 2022 ~ 2023 |
| 5 | 2000 ~ 2021 | 2022 ~ 2023 | 2024 ~ 2025 |

Out-of-sample evaluation period: **2016 ~ 2025** (test periods concatenated)

### Sample Construction

- **Rebalancing** : monthly (last trading day of each month)
- **Lookback** : 2 calendar years → node features + adjacency matrix
- **Horizon** : 1 calendar month → `r_next` (non-overlapping, consistent with monthly rebalancing)

---

## Statistical Testing Design

Two-layer paired t-test structure (one-sided, N seeds):

### Layer 1 — Proposed Models vs Baselines
Bonferroni correction applied (4 comparisons → α = 0.05 / 4 = **0.0125**)

| Pair | Comparison |
|---|---|
| L1-1 | `Graph-L1` vs `EW` |
| L1-2 | `Graph-L1` vs `SampleGMV` |
| L1-3 | `Graph-L2` vs `EW` |
| L1-4 | `Graph-L2` vs `SampleGMV` |

### Layer 2 — RQ-Specific Comparisons
No correction (α = 0.05); each pair isolates a single design choice.

| RQ | Pair | Isolated Effect |
|---|---|---|
| RQ2a | `Graph-L1` vs `Graph-L2` | Loss function (Graph arch fixed) |
| RQ2b | `MLP-L1` vs `MLP-L2` | Loss function (MLP arch fixed) — consistency check |
| RQ3a | `Graph-L1` vs `MLP-L1` | Graph structure (L1 loss fixed) |
| RQ3b | `Graph-L2` vs `MLP-L2` | Graph structure (L2 loss fixed) — consistency check |

---

## Repository Structure

```
├── experiment.ipynb      # Main: training, evaluation, multi-seed CSV export
├── stats_test.ipynb      # Statistical testing: load CSV → Paired t-test + plots
├── models.py             # GraphGMVModel, MLPGMVModel, DifferentiableGMV
├── dataset.py            # PortfolioDataset, feature/adjacency computation
├── losses.py             # loss_variance (L1), loss_mean_variance (L2)
├── FINDINGS.md           # Key empirical findings and interpretations
├── csv/
│   └── 30_industry.csv   # Input data
└── results/              # Generated after running experiment.ipynb (Section 9)
    ├── experiment_results.csv   # seed × strategy × metrics
    ├── ttest_results.csv        # layer × pair × metric × t-stat / p-value
    ├── stat_distributions.png   # Violin plots (metric distributions across seeds)
    └── pvalue_heatmap.png       # p-value heatmap (Layer 1 / Layer 2)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torch-geometric cvxpy cvxpylayers scipy pandas matplotlib networkx
```

### 2. Run experiments

Open `experiment.ipynb` and run all cells top to bottom.

- **Sections 1–8** : single run (seed=42) — training, full-period backtest, visualization
- **Section 9** : N-seed loop (default N=30) → saves `results/experiment_results.csv`

> Approximate runtime: ~9 min per seed on CPU (4 models × 5 folds). 30 seeds ≈ 4–6 hours.

### 3. Statistical testing

Open `stats_test.ipynb` and run all cells.

Requires `results/experiment_results.csv` from step 2.

Outputs:
- Descriptive statistics table (mean ± std per strategy)
- 2-layer paired t-test results (Layer 1: Bonferroni / Layer 2: RQ-specific)
- `results/ttest_results.csv`
- `results/stat_distributions.png` — metric distributions across seeds
- `results/pvalue_heatmap.png` — p-value heatmap (Layer 1 | Layer 2)

---

## Configuration

Key parameters in `experiment.ipynb` → `CFG`:

| Parameter | Default | Description |
|---|---|---|
| `lookback_years` | 2 | Feature window length |
| `horizon_months` | 1 | Holding period (= rebalancing frequency) |
| `corr_threshold` | 0.3 | Adjacency edge threshold (\|Pearson\|) |
| `d_hidden` | 32 | GCN / MLP hidden dimension |
| `d_latent` | 16 | Latent embedding dimension |
| `lr` | 1e-3 | Adam learning rate |
| `n_epochs` | 500 | Max training epochs |
| `patience` | 50 | Early stopping patience |
| `lam_mv` | 1.0 | λ for $\mathcal{L}_2$ (mean-variance trade-off) |

---

## Evaluation Metrics

| Metric | Formula |
|---|---|
| Ann. Return | $\bar{r}_p \times 12$ |
| Ann. Vol | $\sigma_p \times \sqrt{12}$ |
| Sharpe | Ann. Return / Ann. Vol |
| Sortino | Ann. Return / Downside Dev. $\times \sqrt{12}$ |
| Max DD | $\min_t \frac{\text{cum}(t) - \text{peak}(t)}{\text{peak}(t)}$ |
| CVaR (95%) | $-\mathbb{E}[r_p \mid r_p \leq \text{VaR}_{0.05}]$ | 
