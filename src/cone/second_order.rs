//! The second-order cone `{(t, x) : ||x|| <= t}`, which is self-dual, with
//! barrier `F(s) = -log(s_0^2 - ||s_1||^2) / 2` and barrier parameter 1.
//!
//! The scaling is Nesterov–Todd, following Clarabel's `socone.rs`: `W = W'`
//! with `W y = W^{-1} s = lambda`, stored as a scalar `eta` and a vector `w`
//! with `w' J w = 1`, so that `H = W^2 = eta^2 (2 w w' - J)` for
//! `J = diag(1, -I)`. In the Jordan algebra with identity `e = (1, 0)`,
//! `x o y = (x'y, x_0 y_1 + y_0 x_1)` and `x^{-1} = J x / (x' J x)`.

use super::{BlockScaling, ConeGeometryError, ConeKernel, ConeSide, ScalingShape};

/// Blocks of at most this dimension store `H` densely in the KKT matrix, as
/// in Clarabel; larger ones use a rank-2 sparse expansion.
const DENSE_MAX_DIM: usize = 4;

/// A second-order cone block of dimension at least 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SecondOrderCone {
    dim: usize,
}

impl SecondOrderCone {
    pub(super) fn new(dim: usize) -> Self {
        debug_assert!(dim >= 2);
        Self { dim }
    }

    fn is_dense(&self) -> bool {
        self.dim <= DENSE_MAX_DIM
    }

    /// Write the stored diagonal `D` of `H` and the data `(eta, w, lambda)`.
    fn write_scaling(
        &self,
        eta: f64,
        w: &[f64],
        lambda: &[f64],
        diagonal: &mut [f64],
        data: &mut [f64],
    ) {
        let eta2 = eta * eta;
        if self.is_dense() {
            // diag(eta^2 (2 w w' - J)); 2 w_0^2 - 1 is factored to avoid cancellation.
            diagonal[0] = eta2
                * ((std::f64::consts::SQRT_2 * w[0] - 1.0)
                    * (std::f64::consts::SQRT_2 * w[0] + 1.0));
            for (d, &w) in diagonal[1..].iter_mut().zip(&w[1..]) {
                *d = eta2 * (2.0 * w * w + 1.0);
            }
        } else {
            // D = eta^2 I in H = eta^2 (I + u u' - v v').
            diagonal.fill(eta2);
        }
        let (scalar, vectors) = data.split_at_mut(1);
        scalar[0] = eta;
        let (w_out, lambda_out) = vectors.split_at_mut(self.dim);
        w_out.copy_from_slice(w);
        lambda_out.copy_from_slice(lambda);
    }
}

/// The data of one block's scaling: `eta`, `w` and `lambda`.
struct Nt<'a> {
    eta: f64,
    w: &'a [f64],
    lambda: &'a [f64],
}

impl<'a> Nt<'a> {
    fn new(data: &'a [f64]) -> Self {
        let (vectors, dim) = (&data[1..], (data.len() - 1) / 2);
        Self {
            eta: data[0],
            w: &vectors[..dim],
            lambda: &vectors[dim..],
        }
    }

    /// `out = W x` by the ECOS formula.
    fn mul_w(&self, x: &[f64], out: &mut [f64]) {
        let zeta = dot(&self.w[1..], &x[1..]);
        let c = x[0] + zeta / (1.0 + self.w[0]);
        out[0] = self.eta * (self.w[0] * x[0] + zeta);
        for ((out, &w), &x) in out[1..].iter_mut().zip(&self.w[1..]).zip(&x[1..]) {
            *out = self.eta * (c * w + x);
        }
    }

    /// `out = W^{-1} x` by the ECOS formula.
    fn mul_w_inverse(&self, x: &[f64], out: &mut [f64]) {
        let zeta = dot(&self.w[1..], &x[1..]);
        let c = -x[0] + zeta / (1.0 + self.w[0]);
        out[0] = (self.w[0] * x[0] - zeta) / self.eta;
        for ((out, &w), &x) in out[1..].iter_mut().zip(&self.w[1..]).zip(&x[1..]) {
            *out = (c * w + x) / self.eta;
        }
    }
}

/// The vectors `u` and `v` of `H = eta^2 (I + u u' - v v')`.
///
/// Clarabel instead writes `H = eta^2 (D + u u' - v v')` with
/// `D = diag(d, 1, ..., 1)`, which keeps `D - v v'` positive definite only
/// for a corner `d < 1 / ||w||^2`. Near the cone boundary `||w||^2` grows
/// like `1 / mu`, and an unpivoted LDL' that eliminates that tiny corner
/// before the expansion columns loses all accuracy; Clarabel relies on
/// static regularization and refinement of the expanded system instead.
///
/// Here, with `r = ||w_1||` and `w_hat = (0, w_1 / r)`, `-J = I - 2 e e'`
/// gives `H / eta^2 - I = 2 w w' - 2 e e'`, which acts on
/// `span{e, w_hat}` as `2 r [r, w_0; w_0, r]` with eigenvalues `2r(r + w_0)`
/// on `e + w_hat` and `-2r / (w_0 + r)` on `e - w_hat`, using
/// `w_0^2 - 1 = r^2`. So `u = sqrt(r (r + w_0)) (e + w_hat)` and
/// `v = sqrt(r / (w_0 + r)) (e - w_hat)`. Every stored diagonal is `eta^2`,
/// and `I - v v'` stays positive definite since
/// `||v||^2 = 2r / (w_0 + r) < 1`; the system's small eigenvalue
/// `1 / (w_0 + r)^2` appears only in the final pivot, as for a dense block.
fn expansion_vectors(w: &[f64], u: &mut [f64], v: &mut [f64]) {
    let r = norm(&w[1..]);
    if r == 0.0 {
        u.fill(0.0);
        v.fill(0.0);
        return;
    }
    let sum = w[0] + r;
    u[0] = (r * sum).sqrt();
    v[0] = (r / sum).sqrt();
    // The w_hat components, without forming w_1 / r for tiny r.
    let (u_scale, v_scale) = ((sum / r).sqrt(), -(r * sum).sqrt().recip());
    for ((u, v), &w) in u[1..].iter_mut().zip(&mut v[1..]).zip(&w[1..]) {
        *u = u_scale * w;
        *v = v_scale * w;
    }
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}

fn norm(x: &[f64]) -> f64 {
    faer::ColRef::from_slice(x).norm_l2()
}

/// `x_0^2 - ||x_1||^2`, factored to avoid cancellation.
fn residual(x: &[f64]) -> f64 {
    let n = norm(&x[1..]);
    (x[0] - n) * (x[0] + n)
}

fn sqrt_residual(x: &[f64]) -> f64 {
    let r = residual(x);
    if r > 0.0 { r.sqrt() } else { 0.0 }
}

fn is_interior(v: &[f64]) -> bool {
    v.iter().all(|value| value.is_finite()) && v[0] > 0.0 && residual(v) > 0.0
}

/// Project `v` onto the second-order cone in place.
fn project(v: &mut [f64]) {
    let n = norm(&v[1..]);
    if n <= v[0] {
        return;
    }
    if n <= -v[0] {
        v.fill(0.0);
        return;
    }
    let scale = 0.5 * (v[0] + n);
    v[0] = scale;
    for value in &mut v[1..] {
        *value *= scale / n;
    }
}

/// Raise `v_0` so that `v_0 - ||v_1|| >= floor`, and strictly interior even
/// when the floor is below the rounding of `||v_1||`.
fn floor_margin(v: &mut [f64], floor: f64) {
    let n = norm(&v[1..]);
    let margin = v[0] - n;
    if margin.is_nan() || margin < floor {
        v[0] = n + floor;
    }
    if !(residual(v) > 0.0 && v[0] > 0.0) {
        v[0] = v[0].max(n.next_up());
    }
}

/// `x o y`.
fn circ(x: &[f64], y: &[f64], out: &mut [f64]) {
    out[0] = dot(x, y);
    let (x0, y0) = (x[0], y[0]);
    for ((out, &xi), &yi) in out[1..].iter_mut().zip(&x[1..]).zip(&y[1..]) {
        *out = x0 * yi + y0 * xi;
    }
}

/// The largest `alpha` in `[0, 1]` with `x + alpha dx` in the cone, for
/// an interior `x`, following Clarabel's cancellation-safe roots.
fn step_length(x: &[f64], dx: &[f64]) -> f64 {
    let mut alpha_max: f64 = 1.0;
    if x[0] >= 0.0 && dx[0] < 0.0 {
        alpha_max = alpha_max.min(-x[0] / dx[0]);
    }
    // Smallest positive root of ||x_1 + a dx_1||^2 = (x_0 + a dx_0)^2.
    let a = residual(dx);
    let b = 2.0 * (x[0] * dx[0] - dot(&x[1..], &dx[1..]));
    let c = residual(x).max(0.0);
    let d = b * b - 4.0 * a * c;
    if (a > 0.0 && b > 0.0) || d < 0.0 || a == 0.0 {
        return alpha_max;
    }
    if c == 0.0 {
        return if a >= 0.0 { alpha_max } else { 0.0 };
    }
    let t = if b >= 0.0 {
        -b - d.sqrt()
    } else {
        -b + d.sqrt()
    };
    let positive = |root: f64| if root < 0.0 { f64::INFINITY } else { root };
    alpha_max.min(positive(2.0 * c / t).min(positive(t / (2.0 * a))))
}

impl ConeKernel for SecondOrderCone {
    fn barrier_parameter(&self) -> usize {
        1
    }

    fn scaling_shape(&self) -> ScalingShape {
        if self.is_dense() {
            ScalingShape::Dense
        } else {
            ScalingShape::Expanded { rank: 2 }
        }
    }

    fn max_step_fraction(&self) -> f64 {
        // Clarabel's step fraction. Stepping closer to the boundary shrinks
        // one eigenvalue of s or y, and the cone's whole NT scaling, by the
        // same factor: the next scaling is badly centred and later steps
        // stay short (67 iterations against 14 on a medium random SOCP).
        0.99
    }

    fn rectify_row_scaling(&self, scales: &mut [f64]) {
        // The cone is invariant only under a common positive scale.
        let mean = scales.iter().sum::<f64>() / scales.len() as f64;
        scales.fill(mean);
    }

    fn is_primal_interior(&self, s: &[f64]) -> bool {
        is_interior(s)
    }

    fn is_dual_interior(&self, y: &[f64]) -> bool {
        is_interior(y)
    }

    fn unit_point(&self, out: &mut [f64]) {
        out.fill(0.0);
        out[0] = 1.0;
    }

    fn identity_scaling(&self, diagonal: &mut [f64], data: &mut [f64]) {
        let mut e = vec![0.0; self.dim];
        e[0] = 1.0;
        self.write_scaling(1.0, &e, &e, diagonal, data);
    }

    fn margin(&self, v: &[f64]) -> f64 {
        v[0] - norm(&v[1..])
    }

    fn unit_shift(&self, v: &mut [f64], alpha: f64, _side: ConeSide) {
        v[0] += alpha;
    }

    fn reconstruct_slack(&self, b: &[f64], ax: &[f64], out: &mut [f64]) {
        for ((out, b), ax) in out.iter_mut().zip(b).zip(ax) {
            *out = b - ax;
        }
        project(out);
    }

    fn project_dual(&self, y: &mut [f64]) {
        project(y);
    }

    fn floor_interior(&self, s: &mut [f64], y: &mut [f64], floor: f64) {
        floor_margin(s, floor);
        floor_margin(y, floor);
    }

    fn scaling_len(&self) -> usize {
        1 + 2 * self.dim
    }

    fn update_scaling(&self, s: &[f64], y: &[f64], diagonal: &mut [f64], data: &mut [f64]) {
        let y_scale = sqrt_residual(y);
        let s_scale = sqrt_residual(s);
        let eta = (s_scale / y_scale).sqrt();
        // w = normalize(s / s_scale + J y / y_scale).
        let mut w = s.iter().map(|s| s / s_scale).collect::<Vec<_>>();
        w[0] += y[0] / y_scale;
        for (w, y) in w[1..].iter_mut().zip(&y[1..]) {
            *w -= y / y_scale;
        }
        let w_scale = sqrt_residual(&w);
        for w in &mut w {
            *w /= w_scale;
        }
        // Force w' J w = 1 despite rounding.
        w[0] = (1.0 + dot(&w[1..], &w[1..])).sqrt();
        // lambda = W y = W^{-1} s, in closed form.
        let gamma = 0.5 * w_scale;
        let mut lambda = vec![0.0; self.dim];
        lambda[0] = gamma;
        let (cs, cy) = (
            (gamma + y[0] / y_scale) / s_scale,
            (gamma + s[0] / s_scale) / y_scale,
        );
        let denominator = s[0] / s_scale + y[0] / y_scale + 2.0 * gamma;
        for ((lambda, s), y) in lambda[1..].iter_mut().zip(&s[1..]).zip(&y[1..]) {
            *lambda = (cs * s + cy * y) / denominator;
        }
        let root = (s_scale * y_scale).sqrt();
        for lambda in &mut lambda {
            *lambda *= root;
        }
        self.write_scaling(eta, &w, &lambda, diagonal, data);
    }

    fn mul_scaling(&self, scaling: BlockScaling<'_>, v: &[f64], out: &mut [f64]) {
        let nt = Nt::new(scaling.data);
        let c = 2.0 * dot(nt.w, v);
        let eta2 = nt.eta * nt.eta;
        out[0] = eta2 * c.mul_add(nt.w[0], -v[0]);
        for ((out, &w), &v) in out[1..].iter_mut().zip(&nt.w[1..]).zip(&v[1..]) {
            *out = eta2 * c.mul_add(w, v);
        }
    }

    fn abs_mul_scaling(&self, scaling: BlockScaling<'_>, v: &[f64], out: &mut [f64]) {
        // |H| = eta^2 (2 |w| |w|' + I) off the corner, and H_00 =
        // eta^2 (2 w_0^2 - 1) > 0 since w_0 >= 1, so this is exact.
        let nt = Nt::new(scaling.data);
        let c = 2.0 * nt.w.iter().zip(v).map(|(w, v)| (w * v).abs()).sum::<f64>();
        let eta2 = nt.eta * nt.eta;
        out[0] = eta2 * (c * nt.w[0].abs() - v[0].abs());
        for ((out, &w), &v) in out[1..].iter_mut().zip(&nt.w[1..]).zip(&v[1..]) {
            *out = eta2 * (c * w.abs() + v.abs());
        }
    }

    fn write_kkt_coupling(
        &self,
        scaling: BlockScaling<'_>,
        off_diagonal: &mut [f64],
        expansion_diagonal: &mut [f64],
    ) {
        let nt = Nt::new(scaling.data);
        let eta2 = nt.eta * nt.eta;
        if self.is_dense() {
            let mut next = 0;
            for j in 1..self.dim {
                for i in 0..j {
                    off_diagonal[next] = -2.0 * eta2 * nt.w[i] * nt.w[j];
                    next += 1;
                }
            }
            return;
        }
        // As in Clarabel's layout: the first column holds -eta^2 v with
        // diagonal -eta^2, the second -eta^2 u with diagonal eta^2, so
        // eliminating them adds eta^2 (u u' - v v') to the stored eta^2 I.
        let (v, u) = off_diagonal.split_at_mut(self.dim);
        expansion_vectors(nt.w, u, v);
        for value in off_diagonal.iter_mut() {
            *value *= -eta2;
        }
        expansion_diagonal.copy_from_slice(&[-eta2, eta2]);
    }

    fn centering_target(&self, y: &[f64], mu: f64, out: &mut [f64]) {
        let scale = mu / residual(y);
        out[0] = scale * y[0];
        for (out, y) in out[1..].iter_mut().zip(&y[1..]) {
            *out = -scale * y;
        }
    }

    fn correction(
        &self,
        scaling: BlockScaling<'_>,
        y: &[f64],
        ds: &[f64],
        dy: &[f64],
        out: &mut [f64],
    ) {
        let nt = Nt::new(scaling.data);
        let dim = self.dim;
        // shift = (W^{-1} ds) o (W dy), the affine complementarity residual.
        let (mut scaled_ds, mut scaled_dy, mut shift) =
            (vec![0.0; dim], vec![0.0; dim], vec![0.0; dim]);
        nt.mul_w_inverse(ds, &mut scaled_ds);
        nt.mul_w(dy, &mut scaled_dy);
        circ(&scaled_ds, &scaled_dy, &mut shift);
        // out = -W (lambda \ shift), by Clarabel's more stable rearrangement
        // in terms of y.
        let (lambda, w, eta) = (nt.lambda, nt.w, nt.eta);
        let lambda1_shift1 = dot(&lambda[1..], &shift[1..]);
        let w1_shift1 = dot(&w[1..], &shift[1..]);
        let c = (lambda[0] * shift[0] - lambda1_shift1) / residual(y);
        out[0] = c * y[0] + eta * w1_shift1;
        let ratio = w1_shift1 / (1.0 + w[0]);
        for (((out, &yi), &shift_i), &wi) in out[1..]
            .iter_mut()
            .zip(&y[1..])
            .zip(&shift[1..])
            .zip(&w[1..])
        {
            *out = -c * yi + eta * (shift_i + ratio * wi);
        }
        for out in out.iter_mut() {
            *out = -*out / lambda[0];
        }
    }

    fn step_length(&self, v: &[f64], dv: &[f64]) -> Result<f64, ConeGeometryError> {
        if !dv.iter().all(|value| value.is_finite()) {
            return Err(ConeGeometryError::NonFiniteInput { argument: "dv" });
        }
        Ok(step_length(v, dv))
    }

    fn accumulate_local_score(&self, y: &[f64], mu: f64, residual_out: &mut [f64], sum: &mut f64) {
        // r <- r - mu y^{-1}, then add r' (mu (I + P(y)^{-1}))^{-1} r using
        // the eigenvalues of P(y): lambda_k^2 on (1, +-y_1 / ||y_1||) / sqrt(2)
        // for lambda = y_0 +- ||y_1||, and det(y) on the complement.
        let mut target = vec![0.0; self.dim];
        self.centering_target(y, mu, &mut target);
        for (r, t) in residual_out.iter_mut().zip(&target) {
            *r -= t;
        }
        let r = &*residual_out;
        let n = norm(&y[1..]);
        let (l1, l2) = (y[0] + n, y[0] - n);
        let det = l1 * l2;
        let along = if n > 0.0 {
            dot(&y[1..], &r[1..]) / n
        } else {
            0.0
        };
        let r1sq = dot(&r[1..], &r[1..]);
        let perpendicular = (r1sq - along * along).max(0.0);
        let weight = |eigenvalue: f64| eigenvalue / (eigenvalue + 1.0);
        let plus = 0.5 * (r[0] + along) * (r[0] + along);
        let minus = 0.5 * (r[0] - along) * (r[0] - along);
        *sum +=
            (plus * weight(l1 * l1) + minus * weight(l2 * l2) + perpendicular * weight(det)) / mu;
    }
}

#[cfg(test)]
#[allow(
    clippy::needless_range_loop,
    reason = "dense reference formulas read most clearly with indices"
)]
mod tests {
    use super::*;

    fn dense_h(eta: f64, w: &[f64]) -> Vec<Vec<f64>> {
        let n = w.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let j_ij = if i != j {
                            0.0
                        } else if i == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        eta * eta * (2.0 * w[i] * w[j] - j_ij)
                    })
                    .collect()
            })
            .collect()
    }

    fn mul(matrix: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        matrix.iter().map(|row| dot(row, v)).collect()
    }

    fn inverse(x: &[f64]) -> Vec<f64> {
        let r = residual(x);
        let mut out = x.iter().map(|x| -x / r).collect::<Vec<_>>();
        out[0] = x[0] / r;
        out
    }

    fn assert_close(a: &[f64], b: &[f64], tolerance: f64) {
        for (a, b) in a.iter().zip(b) {
            assert!(
                (a - b).abs() <= tolerance * (1.0 + b.abs()),
                "{a:?} vs {b:?}"
            );
        }
    }

    /// Interior pairs of dimensions spanning the dense and expanded layouts.
    fn pairs() -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![
            (vec![2.0, 1.0], vec![1.5, -0.5]),
            (vec![3.0, 1.0, -2.0], vec![1.0, 0.2, 0.3]),
            (vec![1.0, 0.1, 0.2, -0.3], vec![5.0, -4.0, 1.0, 2.0]),
            (
                vec![4.0, 1.0, -1.0, 2.0, 0.5, -3.0],
                vec![1.0, 0.3, 0.1, -0.2, 0.4, 0.0],
            ),
            (vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![2.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    }

    fn scaling_for(s: &[f64], y: &[f64]) -> (SecondOrderCone, Vec<f64>, Vec<f64>) {
        let cone = SecondOrderCone::new(s.len());
        let mut diagonal = vec![0.0; s.len()];
        let mut data = vec![0.0; cone.scaling_len()];
        cone.update_scaling(s, y, &mut diagonal, &mut data);
        (cone, diagonal, data)
    }

    #[test]
    fn nesterov_todd_scaling_satisfies_both_identities() {
        for (s, y) in pairs() {
            let (cone, diagonal, data) = scaling_for(&s, &y);
            let block = BlockScaling {
                diagonal: &diagonal,
                data: &data,
            };
            let nt = Nt::new(&data);
            let h = dense_h(nt.eta, nt.w);
            // H y = s and H s^{-1} = y^{-1}, through both the dense matrix
            // and the matrix-free product.
            assert_close(&mul(&h, &y), &s, 1e-12);
            assert_close(&mul(&h, &inverse(&s)), &inverse(&y), 1e-12);
            let mut product = vec![0.0; s.len()];
            cone.mul_scaling(block, &y, &mut product);
            assert_close(&product, &s, 1e-12);
            // lambda = W y = W^{-1} s.
            let mut wy = vec![0.0; s.len()];
            nt.mul_w(&y, &mut wy);
            assert_close(&wy, nt.lambda, 1e-12);
            nt.mul_w_inverse(&s, &mut wy);
            assert_close(&wy, nt.lambda, 1e-12);
            // |H| |v| is exact.
            let v = (0..s.len()).map(|i| (i as f64) - 1.5).collect::<Vec<_>>();
            cone.abs_mul_scaling(block, &v, &mut product);
            let exact = h
                .iter()
                .map(|row| row.iter().zip(&v).map(|(h, v)| (h * v).abs()).sum::<f64>())
                .collect::<Vec<_>>();
            assert_close(&product, &exact, 1e-12);
        }
    }

    #[test]
    fn kkt_blocks_reproduce_the_scaling() {
        for (s, y) in pairs() {
            let (cone, diagonal, data) = scaling_for(&s, &y);
            let block = BlockScaling {
                diagonal: &diagonal,
                data: &data,
            };
            let nt = Nt::new(&data);
            let h = dense_h(nt.eta, nt.w);
            let n = s.len();
            let (off, expansion) = cone.scaling_shape().coupling_len(n);
            let (mut off_diagonal, mut expansion_diagonal) = (vec![0.0; off], vec![0.0; expansion]);
            cone.write_kkt_coupling(block, &mut off_diagonal, &mut expansion_diagonal);
            // Rebuild H from the stored entries: -(D + E G^{-1} E') = -H.
            let mut rebuilt = vec![vec![0.0; n]; n];
            for i in 0..n {
                rebuilt[i][i] = diagonal[i];
            }
            match cone.scaling_shape() {
                ScalingShape::Dense => {
                    let mut next = 0;
                    for j in 1..n {
                        for i in 0..j {
                            rebuilt[i][j] = -off_diagonal[next];
                            rebuilt[j][i] = -off_diagonal[next];
                            next += 1;
                        }
                    }
                }
                ScalingShape::Expanded { rank } => {
                    assert_eq!(rank, 2);
                    for k in 0..rank {
                        let e = &off_diagonal[k * n..(k + 1) * n];
                        for i in 0..n {
                            for j in 0..n {
                                rebuilt[i][j] += e[i] * e[j] / expansion_diagonal[k];
                            }
                        }
                    }
                    // Quasi-definiteness: D - v v' (the negative expansion
                    // column alone) is positive definite, checked on its
                    // diagonal-dominant Schur form via Cholesky.
                    let e = &off_diagonal[..n];
                    let mut m = vec![vec![0.0; n]; n];
                    for i in 0..n {
                        for j in 0..n {
                            m[i][j] = e[i] * e[j] / expansion_diagonal[0];
                        }
                        m[i][i] += diagonal[i];
                    }
                    assert!(cholesky_succeeds(m));
                }
                _ => unreachable!(),
            }
            for (row, expected) in rebuilt.iter().zip(&h) {
                assert_close(row, expected, 1e-12);
            }
            if n <= DENSE_MAX_DIM {
                assert_eq!(cone.scaling_shape(), ScalingShape::Dense);
                for i in 0..n {
                    assert!((diagonal[i] - h[i][i]).abs() <= 1e-12 * h[i][i]);
                }
            }
        }
    }

    fn cholesky_succeeds(mut m: Vec<Vec<f64>>) -> bool {
        let n = m.len();
        for k in 0..n {
            if m[k][k] <= 0.0 {
                return false;
            }
            let pivot = m[k][k].sqrt();
            for i in k..n {
                m[i][k] /= pivot;
            }
            for j in k + 1..n {
                for i in j..n {
                    m[i][j] -= m[i][k] * m[j][k];
                }
            }
        }
        true
    }

    #[test]
    fn identity_scaling_is_the_identity_in_both_layouts() {
        for n in [2, 4, 5, 7] {
            let cone = SecondOrderCone::new(n);
            let mut diagonal = vec![0.0; n];
            let mut data = vec![0.0; cone.scaling_len()];
            cone.identity_scaling(&mut diagonal, &mut data);
            let block = BlockScaling {
                diagonal: &diagonal,
                data: &data,
            };
            let v = (0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>();
            let mut product = vec![0.0; n];
            cone.mul_scaling(block, &v, &mut product);
            assert_close(&product, &v, 1e-15);
        }
    }

    #[test]
    fn correction_matches_the_jordan_algebra_definition() {
        for (s, y) in pairs() {
            let (cone, diagonal, data) = scaling_for(&s, &y);
            let block = BlockScaling {
                diagonal: &diagonal,
                data: &data,
            };
            let nt = Nt::new(&data);
            let n = s.len();
            let ds = (0..n).map(|i| 0.3 * i as f64 - 0.7).collect::<Vec<_>>();
            let dy = (0..n)
                .map(|i| 0.5 - 0.2 * (i * i) as f64)
                .collect::<Vec<_>>();
            let mut out = vec![0.0; n];
            cone.correction(block, &y, &ds, &dy, &mut out);
            // -W (lambda \ ((W^{-1} ds) o (W dy))), with lambda \ x the
            // solution z of lambda o z = x, i.e. Arw(lambda)^{-1} x.
            let (mut a, mut b, mut shift) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            nt.mul_w_inverse(&ds, &mut a);
            nt.mul_w(&dy, &mut b);
            circ(&a, &b, &mut shift);
            let lambda = nt.lambda;
            let r = residual(lambda);
            let v = dot(&lambda[1..], &shift[1..]);
            let mut z = vec![0.0; n];
            z[0] = (lambda[0] * shift[0] - v) / r;
            let c1 = (v / lambda[0] - shift[0]) / r;
            for i in 1..n {
                z[i] = c1 * lambda[i] + shift[i] / lambda[0];
            }
            let mut check = vec![0.0; n];
            circ(lambda, &z, &mut check);
            assert_close(&check, &shift, 1e-12);
            let mut expected = vec![0.0; n];
            nt.mul_w(&z, &mut expected);
            for value in &mut expected {
                *value = -*value;
            }
            assert_close(&out, &expected, 1e-11);
        }
    }

    #[test]
    fn centering_target_is_the_jordan_inverse() {
        for (_, y) in pairs() {
            let cone = SecondOrderCone::new(y.len());
            let mut out = vec![0.0; y.len()];
            cone.centering_target(&y, 2.0, &mut out);
            let mut product = vec![0.0; y.len()];
            circ(&y, &out, &mut product);
            let mut expected = vec![0.0; y.len()];
            expected[0] = 2.0;
            assert_close(&product, &expected, 1e-14);
        }
    }

    #[test]
    fn step_length_reaches_the_boundary() {
        let x = [2.0, 1.0, 0.0];
        // Along -x the boundary is the origin, beyond the unit step.
        assert_eq!(step_length(&x, &[-1.0, -0.5, 0.0]), 1.0);
        // Moving x_1 by +2 reaches ||x_1|| = x_0 after alpha = 1/2.
        assert!((step_length(&x, &[0.0, 2.0, 0.0]) - 0.5).abs() < 1e-15);
        // A shrinking x_0 meets the boundary where 2 - 2a = 1.
        let alpha = step_length(&x, &[-2.0, 0.0, 0.0]);
        assert!((alpha - 0.5).abs() < 1e-15);
        // Every root is checked against the boundary condition.
        for (s, y) in pairs() {
            let dv = y.iter().map(|v| -3.0 * v).collect::<Vec<_>>();
            let alpha = step_length(&s, &dv);
            let point = s
                .iter()
                .zip(&dv)
                .map(|(s, d)| s + alpha * d)
                .collect::<Vec<_>>();
            if alpha < 1.0 {
                assert!(residual(&point).abs() < 1e-10 * (1.0 + dot(&point, &point)));
            } else {
                assert!(residual(&point) >= 0.0);
            }
        }
    }

    #[test]
    fn projection_and_flooring_reach_the_cone() {
        let mut inside = vec![2.0, 1.0, 1.0];
        project(&mut inside);
        assert_eq!(inside, [2.0, 1.0, 1.0]);
        let mut polar = vec![-2.0, 1.0, 1.0];
        project(&mut polar);
        assert_eq!(polar, [0.0, 0.0, 0.0]);
        let mut outside = vec![0.0, 3.0, 4.0];
        project(&mut outside);
        assert_close(&outside, &[2.5, 1.5, 2.0], 1e-15);
        for floor in [1e-32, 1e-3] {
            let mut v = vec![5.0, 3.0, 4.0];
            floor_margin(&mut v, floor);
            assert!(is_interior(&v));
            // The margin is the floor, up to the rounding of 5 + floor.
            assert!(v[0] > 5.0 && v[0] - 5.0 <= floor + 1e-15);
        }
    }

    #[test]
    fn local_score_matches_the_dense_barrier_hessian() {
        for (_, y) in pairs() {
            let n = y.len();
            let cone = SecondOrderCone::new(n);
            let mu = 0.7;
            let mut residual_out = (0..n).map(|i| 0.4 - 0.3 * i as f64).collect::<Vec<_>>();
            let initial = residual_out.clone();
            let mut sum = 0.0;
            cone.accumulate_local_score(&y, mu, &mut residual_out, &mut sum);
            // r = r0 - mu y^{-1}; weight mu (I + P(y^{-1})).
            let y_inv = inverse(&y);
            let r = initial
                .iter()
                .zip(&y_inv)
                .map(|(r, t)| r - mu * t)
                .collect::<Vec<_>>();
            assert_close(&residual_out, &r, 1e-14);
            let det = residual(&y);
            let mut m = dense_h(1.0, &y_inv);
            for i in 0..n {
                for j in 0..n {
                    // P(y^{-1}) = 2 y^{-1} y^{-1}' - det(y^{-1}) J.
                    let j_ij = if i != j {
                        0.0
                    } else if i == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    m[i][j] = mu
                        * (2.0 * y_inv[i] * y_inv[j] - j_ij / det + if i == j { 1.0 } else { 0.0 });
                }
            }
            let solution = solve_dense(m, r.clone());
            let expected = dot(&r, &solution);
            assert!(
                (sum - expected).abs() <= 1e-12 * expected,
                "{sum} vs {expected}"
            );
        }
    }

    fn solve_dense(mut m: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for k in 0..n {
            for i in k + 1..n {
                let factor = m[i][k] / m[k][k];
                for j in k..n {
                    m[i][j] -= factor * m[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
        for k in (0..n).rev() {
            for j in k + 1..n {
                b[k] -= m[k][j] * b[j];
            }
            b[k] /= m[k][k];
        }
        b
    }
}
