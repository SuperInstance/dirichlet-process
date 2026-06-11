//! Dirichlet Process
//!
//! Dirichlet process construction via the stick-breaking process and
//! Chinese restaurant process representations.

use std::collections::HashMap;

/// Stick-breaking construction of a Dirichlet process.
pub struct StickBreaking {
    alpha: f64,
    rng: rand::rngs::ThreadRng,
}

impl StickBreaking {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0, "concentration must be positive");
        use rand::Rng;
        Self {
            alpha,
            rng: rand::thread_rng(),
        }
    }

    /// Break the stick into k pieces, returning (weight, cumulative_weight) pairs.
    pub fn break_stick(&mut self, k: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(k);
        let mut remaining = 1.0_f64;
        use rand::Rng;
        for i in 0..k {
            if i == k - 1 {
                weights.push(remaining);
            } else {
                let beta_a = 1.0;
                let beta_b = self.alpha;
                let v = self.sample_beta(beta_a, beta_b);
                let w = remaining * v;
                weights.push(w);
                remaining -= w;
            }
        }
        weights
    }

    fn sample_beta(&mut self, a: f64, b: f64) -> f64 {
        use rand::Rng;
        let x: f64 = self.rng.gen_range(0.0..1.0);
        // Approximate beta via simple rejection; for production use statrs
        // Here we use a gamma-ratio approach with Box-Muller for approximation
        let g1 = self.sample_gamma(a);
        let g2 = self.sample_gamma(b);
        g1 / (g1 + g2)
    }

    fn sample_gamma(&mut self, shape: f64) -> f64 {
        // Marsaglia and Tsang's method for shape >= 1
        use rand::Rng;
        let shape = shape.max(1.0);
        let d = shape - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();
        loop {
            let u1: f64 = self.rng.gen_range(0.0..1.0);
            let u2: f64 = self.rng.gen_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v > 0.0 {
                let u: f64 = self.rng.gen_range(0.0..1.0);
                if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                    return d * v;
                }
            }
        }
    }
}

/// Chinese Restaurant Process representation.
pub struct ChineseRestaurantProcess {
    alpha: f64,
    /// Maps table (cluster) index to customer count.
    tables: HashMap<usize, usize>,
    n_customers: usize,
    n_tables: usize,
}

impl ChineseRestaurantProcess {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            tables: HashMap::new(),
            n_customers: 0,
            n_tables: 0,
        }
    }

    /// Seat a new customer; returns the table index.
    pub fn seat(&mut self) -> usize {
        use rand::Rng;
        let total = self.n_customers as f64 + self.alpha;
        let u: f64 = rand::thread_rng().gen_range(0.0..total);

        let mut cumulative = 0.0;
        for (&table, &count) in &self.tables {
            cumulative += count as f64;
            if u < cumulative {
                self.tables.insert(table, count + 1);
                self.n_customers += 1;
                return table;
            }
        }

        // New table
        let new_table = self.n_tables;
        self.tables.insert(new_table, 1);
        self.n_tables += 1;
        self.n_customers += 1;
        new_table
    }

    /// Get the number of occupied tables.
    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    /// Get the number of customers.
    pub fn num_customers(&self) -> usize {
        self.n_customers
    }

    /// Expected number of tables for n customers.
    pub fn expected_tables(alpha: f64, n: usize) -> f64 {
        (1..=n).map(|i| alpha / (alpha + i as f64)).sum::<f64>() + alpha / (alpha + 0.0_f64).max(1.0)
    }
}

/// Compute the probability of a particular partition under the CRP.
pub fn crp_partition_probability(alpha: f64, table_sizes: &[usize]) -> f64 {
    let n: usize = table_sizes.iter().sum();
    let k = table_sizes.len();
    let mut log_p = 0.0;
    for (m, &size) in table_sizes.iter().enumerate() {
        log_p += (size as f64 - 1.0).max(0.0).ln().max(-1e10);
        log_p += alpha.ln();
        log_p -= (alpha + m as f64).ln();
    }
    // Rising factorial normalization
    for i in 0..n {
        log_p -= (i as f64).ln().max(-1e10);
    }
    log_p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stick_weights_sum_to_one() {
        let mut sb = StickBreaking::new(1.0);
        let w = sb.break_stick(10);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_crp_seating() {
        let mut crp = ChineseRestaurantProcess::new(2.0);
        for _ in 0..20 {
            crp.seat();
        }
        assert_eq!(crp.num_customers(), 20);
        assert!(crp.num_tables() > 0);
    }
}
