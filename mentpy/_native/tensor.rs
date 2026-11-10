// Copyright 2026 Luis Mantilla
// Licensed under the Apache License, Version 2.0.
// See <http://www.apache.org/licenses/LICENSE-2.0> for details.
//! CPU complex contractions with a compact reverse-mode tape.
//!
//! Each contraction has one consumer. Its adjoint is two conjugate matrix
//! products, so we need no scalar graph or recursive traversal. Larger products
//! use gemm's SIMD kernels; tiny graph-state tensors avoid dispatch overhead.

use super::contract::Plan;
use candle_core::Result;
use num_complex::Complex64 as C;

const ZERO: C = C::new(0., 0.);
const ONE: C = C::new(1., 0.);

fn permute(data: Vec<C>, axes: &[usize]) -> Vec<C> {
    if axes.iter().enumerate().all(|(i, &j)| i == j) {
        return data;
    }
    let rank = axes.len();
    let strides: Vec<_> = axes.iter().map(|&a| 1usize << (rank - a - 1)).collect();
    (0..data.len())
        .map(|i| {
            let index = strides.iter().enumerate().fold(0, |index, (j, stride)| {
                index | (((i >> (rank - j - 1)) & 1) * stride)
            });
            data[index]
        })
        .collect()
}

fn undo(data: Vec<C>, axes: &[usize]) -> Vec<C> {
    let mut inverse = vec![0; axes.len()];
    for (i, &j) in axes.iter().enumerate() {
        inverse[j] = i;
    }
    permute(data, &inverse)
}

// Row-major batched matrix multiplication; adjoint flags conjugate-transpose
// the corresponding input without making another copy.
#[allow(clippy::too_many_arguments)]
fn matmul(
    a: &[C],
    b: &[C],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    ah: bool,
    bh: bool,
) -> Vec<C> {
    assert_eq!(a.len(), batch * m * k);
    assert_eq!(b.len(), batch * k * n);
    let mut out = vec![ZERO; batch * m * n];
    let (ars, acs) = if ah { (1, m) } else { (k, 1) };
    let (brs, bcs) = if bh { (1, k) } else { (n, 1) };
    for batch_index in 0..batch {
        let a = &a[batch_index * m * k..(batch_index + 1) * m * k];
        let b = &b[batch_index * k * n..(batch_index + 1) * k * n];
        let out = &mut out[batch_index * m * n..(batch_index + 1) * m * n];
        if m * n * k < 4096 {
            for i in 0..m {
                for j in 0..n {
                    let mut value = ZERO;
                    for p in 0..k {
                        let a = a[i * ars + p * acs];
                        let b = b[p * brs + j * bcs];
                        value += (if ah { a.conj() } else { a }) * (if bh { b.conj() } else { b });
                    }
                    out[i * n + j] = value;
                }
            }
        } else {
            // SAFETY: slice lengths are checked above. The strides address
            // precisely the row-major (or transposed) matrices in these disjoint
            // slices; out owns all m*n writable elements and aliases neither
            // input. num_complex::Complex64 is gemm::c64.
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    out.as_mut_ptr(),
                    1,
                    n as isize,
                    false,
                    a.as_ptr(),
                    acs as isize,
                    ars as isize,
                    b.as_ptr(),
                    bcs as isize,
                    brs as isize,
                    ZERO,
                    ONE,
                    false,
                    ah,
                    bh,
                    gemm::Parallelism::None,
                );
            }
        }
    }
    out
}

struct Saved {
    left: Vec<C>,
    right: Vec<C>,
    scale: f64,
}

pub struct Tape {
    saved: Vec<Saved>,
    n_operands: usize,
}

pub fn contract(plan: &Plan, operands: Vec<Vec<C>>, gradient: bool) -> Result<(Vec<C>, Tape)> {
    let n_operands = operands.len();
    let mut values: Vec<_> = operands.into_iter().map(Some).collect();
    let mut saved = Vec::new();
    for step in &plan.steps {
        let a = permute(values[step.left].take().unwrap(), &step.left_axes);
        let b = permute(values[step.right].take().unwrap(), &step.right_axes);
        let (batch, m, k, n) = (
            step.left_shape[0],
            step.left_shape[1],
            step.left_shape[2],
            step.right_shape[2],
        );
        let mut result = matmul(&a, &b, batch, m, k, n, false, false);
        let scale = result.iter().map(|z| z.norm()).fold(0., f64::max);
        if !scale.is_finite() || scale <= f64::EPSILON {
            candle_core::bail!(
                "Selected measurement branch has zero probability or nonfinite amplitudes."
            );
        }
        for z in &mut result {
            *z /= scale;
        }
        if gradient {
            saved.push(Saved {
                left: a,
                right: b,
                scale,
            });
        }
        values.push(Some(result));
    }
    Ok((
        permute(values.pop().unwrap().unwrap(), &plan.output_axes),
        Tape { saved, n_operands },
    ))
}

impl Tape {
    pub fn backward(self, plan: &Plan, output: Vec<C>) -> Vec<Vec<C>> {
        let mut gradients: Vec<Option<Vec<C>>> = vec![None; self.n_operands + plan.steps.len()];
        *gradients.last_mut().unwrap() = Some(undo(output, &plan.output_axes));
        for (index, (step, saved)) in plan.steps.iter().zip(self.saved).enumerate().rev() {
            let mut grad = gradients[self.n_operands + index].take().unwrap();
            for z in &mut grad {
                *z /= saved.scale;
            }
            let (batch, m, k, n) = (
                step.left_shape[0],
                step.left_shape[1],
                step.left_shape[2],
                step.right_shape[2],
            );
            let a = matmul(&grad, &saved.right, batch, m, n, k, false, true);
            let b = matmul(&saved.left, &grad, batch, k, m, n, true, false);
            gradients[step.left] = Some(undo(a, &step.left_axes));
            gradients[step.right] = Some(undo(b, &step.right_axes));
        }
        gradients
            .into_iter()
            .take(self.n_operands)
            .map(Option::unwrap)
            .collect()
    }
}

pub fn sin_cos(angle: f64) -> (f64, f64) {
    let quarters = angle / std::f64::consts::FRAC_PI_2;
    if quarters.fract() == 0. && quarters.abs() < i64::MAX as f64 {
        let q = (quarters as i64).rem_euclid(4) as usize;
        ([0., 1., 0., -1.][q], [1., 0., -1., 0.][q])
    } else {
        angle.sin_cos()
    }
}

pub fn measurement(theta: f64, latitude: f64) -> [C; 2] {
    let (s, c) = sin_cos(theta);
    let (second, first) = sin_cos(std::f64::consts::FRAC_PI_4 - latitude * 0.5);
    [C::new(c, s) * first, C::new(second, 0.)]
}

pub fn derivative(theta: f64, latitude: f64, grad: &[C]) -> [f64; 2] {
    let (s, c) = sin_cos(theta);
    let phase = C::new(c, s);
    let (second, first) = sin_cos(std::f64::consts::FRAC_PI_4 - latitude * 0.5);
    [
        (grad[0].conj() * C::i() * phase * first).re,
        (grad[0].conj() * phase * (second * 0.5) - grad[1].conj() * (first * 0.5)).re,
    ]
}

pub fn normalize(mut state: Vec<C>) -> Result<(Vec<C>, f64)> {
    let norm = state.iter().map(C::norm_sqr).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 0. {
        candle_core::bail!(
            "Selected measurement branch has zero probability or nonfinite amplitudes."
        );
    }
    for z in &mut state {
        *z /= norm;
    }
    Ok((state, norm))
}

pub fn expectation(
    state: &[C],
    terms: &[(String, f64)],
    constant: f64,
    n_outputs: usize,
) -> Result<(f64, Vec<C>)> {
    let mut action = vec![ZERO; state.len()];
    for (word, coefficient) in terms {
        if word.len() != n_outputs || !word.chars().all(|c| "IXYZ".contains(c)) {
            candle_core::bail!("Observable must contain one Pauli letter per output.");
        }
        let (mut flip, mut sign, mut phase) = (0, 0, ONE);
        for (i, c) in word.chars().enumerate() {
            let bit = 1usize << (n_outputs - i - 1);
            if c == 'X' || c == 'Y' {
                flip |= bit;
            }
            if c == 'Z' || c == 'Y' {
                sign |= bit;
            }
            if c == 'Y' {
                phase *= C::i();
            }
        }
        for (i, z) in state.iter().enumerate() {
            action[i ^ flip] += phase
                * z
                * (if (i & sign).count_ones().is_multiple_of(2) {
                    *coefficient
                } else {
                    -*coefficient
                });
        }
    }
    let energy: f64 = state
        .iter()
        .zip(&action)
        .map(|(s, h)| (s.conj() * h).re)
        .sum();
    // d Re(<psi|H|psi>) / d(raw state), including normalization. Complex
    // adjoints encode the derivatives of real and imaginary parts together.
    let gradient = action
        .into_iter()
        .zip(state)
        .map(|(h, s)| 2. * (h - energy * s))
        .collect();
    Ok((constant + energy, gradient))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_scalar_products_and_adjoints() {
        for (m, k, n) in [(3, 7, 2), (17, 19, 23)] {
            for ah in [false, true] {
                for bh in [false, true] {
                    let a: Vec<_> = (0..2 * m * k)
                        .map(|i| C::new((i as f64).sin(), (i as f64).cos()))
                        .collect();
                    let b: Vec<_> = (0..2 * k * n)
                        .map(|i| C::new((0.7 * i as f64).cos(), (0.2 * i as f64).sin()))
                        .collect();
                    let out = matmul(&a, &b, 2, m, k, n, ah, bh);
                    for batch in 0..2 {
                        for row in 0..m {
                            for col in 0..n {
                                let expected: C = (0..k)
                                    .map(|p| {
                                        let left = a[batch * m * k
                                            + if ah { p * m + row } else { row * k + p }];
                                        let right = b[batch * k * n
                                            + if bh { col * k + p } else { p * n + col }];
                                        (if ah { left.conj() } else { left })
                                            * (if bh { right.conj() } else { right })
                                    })
                                    .sum();
                                assert!(
                                    (out[batch * m * n + row * n + col] - expected).norm() < 1e-11
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
