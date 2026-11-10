// Copyright 2026 Luis Mantilla
// Licensed under the Apache License, Version 2.0.
// See <http://www.apache.org/licenses/LICENSE-2.0> for details.
//! Lazy graph preparation and sampled measurement trajectories.

use super::tensor;
use num_complex::Complex64 as C;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeSet;

#[pyclass]
pub struct Frontier {
    neighbors: Vec<Vec<(usize, usize)>>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    measured: Vec<usize>,
    corrections: Vec<Vec<(usize, u8)>>,
    edges: Vec<(usize, usize)>,
    initial: Option<Vec<C>>,
    state: Vec<C>,
    active: Vec<usize>,
    prepared: Vec<bool>,
    applied: Vec<bool>,
    outcomes: Vec<u8>,
    cursor: usize,
    max_elements: usize,
    max_frontier: usize,
}

#[pymethods]
impl Frontier {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n, edges, inputs, outputs, measured, corrections, max_elements=67108864))]
    fn new(
        n: usize,
        edges: Vec<(usize, usize)>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        measured: Vec<usize>,
        corrections: Vec<Vec<(usize, u8)>>,
        max_elements: usize,
    ) -> PyResult<Self> {
        let unique = |xs: &[usize]| xs.iter().copied().collect::<BTreeSet<_>>();
        if inputs
            .iter()
            .chain(&outputs)
            .chain(&measured)
            .any(|&v| v >= n)
            || unique(&inputs).len() != inputs.len()
            || unique(&outputs).len() != outputs.len()
            || unique(&measured).len() != measured.len()
            || measured.len() + outputs.len() != n
            || !unique(&measured).is_disjoint(&unique(&outputs))
            || corrections.len() != measured.len()
            || corrections.iter().flatten().any(|&(v, p)| v >= n || p > 1)
            || edges.iter().any(|&(u, v)| u >= n || v >= n || u == v)
            || edges
                .iter()
                .map(|&(u, v)| (u.min(v), u.max(v)))
                .collect::<BTreeSet<_>>()
                .len()
                != edges.len()
            || max_elements == 0
        {
            return Err(PyValueError::new_err(
                "Invalid graph, correction, or memory-limit specification.",
            ));
        }
        let mut order = vec![n; n];
        for (i, &v) in measured.iter().enumerate() {
            order[v] = i;
        }
        if corrections
            .iter()
            .enumerate()
            .any(|(i, targets)| targets.iter().any(|&(v, _)| order[v] <= i))
        {
            return Err(PyValueError::new_err(
                "Corrections must target future nodes.",
            ));
        }
        let mut neighbors = vec![vec![]; n];
        for (i, &(u, v)) in edges.iter().enumerate() {
            neighbors[u].push((v, i));
            neighbors[v].push((u, i));
        }
        let mut result = Self {
            neighbors,
            inputs,
            outputs,
            measured,
            corrections,
            applied: vec![false; edges.len()],
            edges,
            initial: None,
            state: vec![],
            active: vec![],
            prepared: vec![false; n],
            outcomes: vec![],
            cursor: 0,
            max_elements,
            max_frontier: 0,
        };
        result.reset();
        Ok(result)
    }

    #[pyo3(signature = (re=None, im=None))]
    fn set_input(&mut self, re: Option<Vec<f64>>, im: Option<Vec<f64>>) -> PyResult<()> {
        self.initial = match (re, im) {
            (None, None) => None,
            (Some(re), Some(im)) => {
                let dimension = 1usize
                    .checked_shl(self.inputs.len() as u32)
                    .ok_or_else(|| PyValueError::new_err("Input dimension overflow."))?;
                if re.len() != dimension || im.len() != dimension || dimension > self.max_elements {
                    return Err(PyValueError::new_err(
                        "Input state dimension exceeds the input size or max_elements.",
                    ));
                }
                let data: Vec<_> = re.into_iter().zip(im).map(|(a, b)| C::new(a, b)).collect();
                let norm: f64 = data.iter().map(C::norm_sqr).sum();
                if !norm.is_finite() || (norm - 1.).abs() > 1e-5 {
                    return Err(PyValueError::new_err("Input state must be normalized."));
                }
                Some(data)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Input real and imaginary parts must be provided together.",
                ))
            }
        };
        self.reset();
        Ok(())
    }

    fn reset(&mut self) {
        self.active.clear();
        self.prepared.fill(false);
        self.applied.fill(false);
        self.outcomes.clear();
        self.cursor = 0;
        if let Some(initial) = &self.initial {
            self.state = initial.clone();
            self.active = self.inputs.clone();
            for &node in &self.inputs {
                self.prepared[node] = true;
            }
        } else {
            self.state = vec![C::new(1., 0.)];
        }
        self.max_frontier = self.active.len();
        for i in 0..self.edges.len() {
            self.entangle(i);
        }
    }

    #[getter]
    fn max_frontier(&self) -> usize {
        self.max_frontier
    }

    #[getter]
    fn outcomes(&self) -> Vec<u8> {
        self.outcomes.clone()
    }

    #[pyo3(signature = (angles, random=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        angles: Vec<f64>,
        random: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if angles.len() != 2 * self.measured.len()
            || random
                .as_ref()
                .is_some_and(|xs| xs.len() != self.measured.len())
        {
            return Err(PyValueError::new_err(
                "Expected one angle pair and optional uniform draw per measured node.",
            ));
        }
        py.allow_threads(|| {
            self.reset();
            for (i, pair) in angles.as_chunks::<2>().0.iter().enumerate() {
                self.measure(pair[0], pair[1], random.as_ref().map(|xs| xs[i]))?;
            }
            self.finish_impl()
        })
    }

    #[pyo3(signature = (theta, latitude, random=None))]
    fn step(
        &mut self,
        py: Python<'_>,
        theta: f64,
        latitude: f64,
        random: Option<f64>,
    ) -> PyResult<u8> {
        py.allow_threads(|| self.measure(theta, latitude, random))
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.allow_threads(|| self.finish_impl())
    }

    fn sample(
        &mut self,
        py: Python<'_>,
        angles: Vec<f64>,
        draws: Vec<f64>,
        basis: &str,
    ) -> PyResult<Vec<u8>> {
        let width = self.measured.len() + 1;
        if angles.len() != 2 * self.measured.len()
            || draws.is_empty()
            || !draws.len().is_multiple_of(width)
            || draws
                .iter()
                .any(|&u| !u.is_finite() || !(0.0..1.0).contains(&u))
            || basis.len() != self.outputs.len()
            || !basis.chars().all(|c| "IXYZ".contains(c))
        {
            return Err(PyValueError::new_err(
                "Invalid angles, random draws, or output basis for sampling.",
            ));
        }
        py.allow_threads(|| {
            let mut samples = Vec::with_capacity(draws.len() / width * self.outputs.len());
            for random in draws.chunks_exact(width) {
                self.reset();
                for (i, pair) in angles.as_chunks::<2>().0.iter().enumerate() {
                    self.measure(pair[0], pair[1], Some(random[i]))?;
                }
                let mut state = self.output_state()?;
                for (axis, pauli) in basis.chars().enumerate() {
                    if pauli == 'X' || pauli == 'Y' {
                        let mask = 1usize << (self.outputs.len() - axis - 1);
                        for i in 0..state.len() {
                            if i & mask == 0 {
                                let a = state[i];
                                let b = state[i | mask]
                                    * (if pauli == 'Y' {
                                        -C::i()
                                    } else {
                                        C::new(1., 0.)
                                    });
                                state[i] = (a + b) * std::f64::consts::FRAC_1_SQRT_2;
                                state[i | mask] = (a - b) * std::f64::consts::FRAC_1_SQRT_2;
                            }
                        }
                    }
                }
                let norm: f64 = state.iter().map(C::norm_sqr).sum();
                if !norm.is_finite() || norm <= 0. {
                    return Err(PyValueError::new_err(
                        "Simulation did not produce valid probabilities.",
                    ));
                }
                let threshold = random[width - 1] * norm;
                let mut cumulative = 0.;
                let index = state
                    .iter()
                    .position(|z| {
                        cumulative += z.norm_sqr();
                        cumulative > threshold
                    })
                    .unwrap_or(state.len() - 1);
                samples.extend(
                    (0..self.outputs.len())
                        .map(|i| ((index >> (self.outputs.len() - i - 1)) & 1) as u8),
                );
            }
            Ok(samples)
        })
    }
}

impl Frontier {
    fn entangle(&mut self, edge: usize) {
        if self.applied[edge] {
            return;
        }
        let (u, v) = self.edges[edge];
        let (Some(a), Some(b)) = (
            self.active.iter().position(|&n| n == u),
            self.active.iter().position(|&n| n == v),
        ) else {
            return;
        };
        let mask =
            (1usize << (self.active.len() - a - 1)) | (1usize << (self.active.len() - b - 1));
        for (i, z) in self.state.iter_mut().enumerate() {
            if i & mask == mask {
                *z = -*z;
            }
        }
        self.applied[edge] = true;
    }

    fn prepare(&mut self, node: usize) -> PyResult<()> {
        if !self.prepared[node] {
            let size = self.state.len().checked_mul(2).filter(|&n| n<=self.max_elements)
                .ok_or_else(|| PyValueError::new_err("Active state exceeds max_elements; use tensor contraction or increase the limit."))?;
            let mut state = Vec::with_capacity(size);
            for &z in &self.state {
                let z = z * std::f64::consts::FRAC_1_SQRT_2;
                state.push(z);
                state.push(z);
            }
            self.state = state;
            self.active.push(node);
            self.prepared[node] = true;
            self.max_frontier = self.max_frontier.max(self.active.len());
        }
        for i in 0..self.neighbors[node].len() {
            self.entangle(self.neighbors[node][i].1);
        }
        Ok(())
    }

    fn correct(&mut self, node: usize, pauli: u8) -> PyResult<()> {
        self.prepare(node)?;
        if pauli == 0 {
            // X must follow every incident CZ; Z commutes with pending CZs.
            for i in 0..self.neighbors[node].len() {
                let neighbor = self.neighbors[node][i].0;
                if !self.prepared[neighbor] {
                    self.prepare(neighbor)?;
                }
            }
        }
        let axis = self.active.iter().position(|&n| n == node).unwrap();
        let mask = 1usize << (self.active.len() - axis - 1);
        if pauli == 0 {
            for i in 0..self.state.len() {
                if i & mask == 0 {
                    self.state.swap(i, i | mask);
                }
            }
        } else {
            for (i, z) in self.state.iter_mut().enumerate() {
                if i & mask != 0 {
                    *z = -*z;
                }
            }
        }
        Ok(())
    }

    fn measure(&mut self, theta: f64, latitude: f64, random: Option<f64>) -> PyResult<u8> {
        if self.cursor == self.measured.len() {
            return Err(PyValueError::new_err(
                "All measurements have already been executed.",
            ));
        }
        if !theta.is_finite()
            || !latitude.is_finite()
            || random.is_some_and(|u| !u.is_finite() || !(0.0..1.0).contains(&u))
        {
            return Err(PyValueError::new_err(
                "Angles must be finite and uniform draws must be in [0, 1).",
            ));
        }
        let node = self.measured[self.cursor];
        self.prepare(node)?;
        for i in 0..self.neighbors[node].len() {
            let neighbor = self.neighbors[node][i].0;
            if !self.prepared[neighbor] {
                self.prepare(neighbor)?;
            }
        }
        let axis = self.active.iter().position(|&n| n == node).unwrap();
        let mask = 1usize << (self.active.len() - axis - 1);
        let first = tensor::measurement(theta, latitude);
        let (sin, cos) = tensor::sin_cos(theta);
        let (second_amp, first_amp) = tensor::sin_cos(std::f64::consts::FRAC_PI_4 - latitude * 0.5);
        let second = [-C::new(cos, sin) * second_amp, C::new(first_amp, 0.)];
        let mut probabilities = [0., 0.];
        for i in 0..self.state.len() / 2 {
            let offset = (i & (mask - 1)) | ((i & !(mask - 1)) << 1);
            let (a, b) = (self.state[offset], self.state[offset | mask]);
            probabilities[0] += (first[0] * a + first[1] * b).norm_sqr();
            if random.is_some() {
                probabilities[1] += (second[0] * a + second[1] * b).norm_sqr();
            }
        }
        let outcome = usize::from(
            random.is_some_and(|u| u >= probabilities[0] / (probabilities[0] + probabilities[1])),
        );
        let probability = probabilities[outcome];
        if !probability.is_finite() || probability <= 1e-28 {
            return Err(PyValueError::new_err(format!(
                "Selected measurement branch has zero probability at node {node}."
            )));
        }
        let vector = if outcome == 0 { first } else { second };
        let norm = probability.sqrt();
        let state = (0..self.state.len() / 2)
            .map(|i| {
                let offset = (i & (mask - 1)) | ((i & !(mask - 1)) << 1);
                (vector[0] * self.state[offset] + vector[1] * self.state[offset | mask]) / norm
            })
            .collect();
        self.state = state;
        self.active.remove(axis);
        self.outcomes.push(outcome as u8);
        if outcome == 1 {
            for i in 0..self.corrections[self.cursor].len() {
                let (target, pauli) = self.corrections[self.cursor][i];
                self.correct(target, pauli)?;
            }
        }
        self.cursor += 1;
        Ok(outcome as u8)
    }

    fn finish_impl(&mut self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        Ok(self
            .output_state()?
            .into_iter()
            .map(|z| (z.re, z.im))
            .unzip())
    }

    fn output_state(&mut self) -> PyResult<Vec<C>> {
        if self.cursor != self.measured.len() {
            return Err(PyValueError::new_err(
                "Execute all measurements before reading the output.",
            ));
        }
        for i in 0..self.outputs.len() {
            self.prepare(self.outputs[i])?;
        }
        let strides: Vec<_> = self
            .outputs
            .iter()
            .map(|node| {
                let axis = self.active.iter().position(|n| n == node).unwrap();
                1usize << (self.active.len() - axis - 1)
            })
            .collect();
        let n = strides.len();
        Ok((0..self.state.len())
            .map(|i| {
                let source = strides.iter().enumerate().fold(0, |index, (j, stride)| {
                    index | (((i >> (n - j - 1)) & 1) * stride)
                });
                self.state[source]
            })
            .collect())
    }
}
