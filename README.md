# Dirichlet Process

**A Bayesian nonparametric library implementing the Dirichlet Process** via the stick-breaking process and Chinese Restaurant Process representations — the foundational mathematical object for infinite mixture models and nonparametric Bayesian clustering.

## Why It Matters

The Dirichlet Process (DP) is one of the most important objects in Bayesian statistics. It's a **distribution over distributions** — a prior probability distribution whose support is itself a space of probability measures. This enables **infinite mixture models** where the number of clusters is not fixed in advance but inferred from data.

**The problem it solves:** In traditional clustering (K-means, Gaussian mixtures), you must choose the number of clusters K beforehand. Pick too few and you merge distinct groups; pick too many and you overfit. The Dirichlet Process solves this by placing a prior over *partitions of the data* that naturally grows with the data, controlled by a single concentration parameter α.

**Real-world applications:**
- **Topic modeling** — The Hierarchical Dirichlet Process (HDP) generalizes LDA to infer the number of topics automatically
- **Clustering** — DP mixture models for customer segmentation, document clustering
- **Genomics** — Inferring population structure without predefined cluster counts
- **NLP** — Infinite hidden Markov models for unsupervised POS tagging

**Two representations implemented:**

1. **Stick-breaking process** (Sethuraman, 1994) — Constructs DP weights by repeatedly breaking a unit-length stick: at step k, break off fraction Vₖ ~ Beta(1, α) of what remains. The resulting weights are a valid probability distribution that sums to 1.

2. **Chinese Restaurant process** (Aldous, 1985) — A metaphor where customers (data points) enter a restaurant and sit at tables (clusters). The probability of sitting at an existing table is proportional to its occupancy; the probability of starting a new table is proportional to α. After n customers, the expected number of tables is O(α log n).

## How It Works

**Stick-breaking construction:** The DP(α, H) is constructed as:
- Sample Vₖ ~ Beta(1, α) for k = 1, 2, ...
- Weights: πₖ = Vₖ · ∏(1 − Vⱼ) for j < k

The library implements this using Marsaglia and Tsang's method for Gamma sampling, which is then ratioed to produce Beta variates. The Gamma sampler uses the acceptance-rejection method with the standard cube-root transform.

**Chinese Restaurant Process:** Each customer either:
- Joins an existing table with probability nₖ / (n + α) where nₖ is the table's current occupancy
- Starts a new table with probability α / (n + α)

This naturally produces a power-law distribution of table sizes — a few large tables, many small ones — which matches real-world cluster distributions.

**Partition probability:** `crp_partition_probability` computes the log-probability of a particular clustering under the CRP prior, useful for inference algorithms like Gibbs sampling.

## Quick Start

```rust
use dirichlet_process::{StickBreaking, ChineseRestaurantProcess};

// Stick-breaking: generate DP weights
let mut sb = StickBreaking::new(1.0); // α = 1 (low concentration → few large clusters)
let weights = sb.break_stick(10);
println!("Weights sum to: {}", weights.iter().sum::<f64>()); // ≈ 1.0

// Chinese Restaurant Process: cluster 100 data points
let mut crp = ChineseRestaurantProcess::new(2.0); // α = 2
for _ in 0..100 {
    let table = crp.seat();
}
println!("{} customers at {} tables", crp.num_customers(), crp.num_tables());
// Typical: 100 customers, ~8-12 tables with α=2

// Expected number of tables for n customers
let expected = ChineseRestaurantProcess::expected_tables(2.0, 100);
```

## API

### `StickBreaking`
- `new(alpha: f64) -> Self` — Create with concentration parameter α
- `break_stick(k: usize) -> Vec<f64>` — Generate k weights summing to 1.0

### `ChineseRestaurantProcess`
- `new(alpha: f64) -> Self` — Create with concentration parameter
- `seat() -> usize` — Add a customer, returns table index
- `num_tables() -> usize` — Current number of occupied tables
- `num_customers() -> usize` — Total customers seated
- `expected_tables(alpha, n) -> f64` — Expected table count for n customers. O(n)

### `crp_partition_probability(alpha, table_sizes) -> f64`
- Log-probability of a partition under the CRP prior

## Architecture Notes

This library provides Bayesian nonparametric primitives for SuperInstance's machine learning toolkit, enabling clustering and mixture modeling without predetermined cluster counts. It supports the inference layer for topic modeling and document clustering pipelines.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## License

MIT
