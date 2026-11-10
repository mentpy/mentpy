// Copyright 2026 Luis Mantilla
// Licensed under the Apache License, Version 2.0.
// See <http://www.apache.org/licenses/LICENSE-2.0> for details.
//! Sparse Heisenberg propagation using PPVM packed words and Clifford rules.
//! The iterative coefficient adjoint retains zero-valued derivative paths.

use fxhash::FxHashMap;
use num_complex::Complex64 as C;
use ppvm_pauli_word::{phase::PhasedPauliWord, word::PauliWord};
use ppvm_traits::traits::{Clifford, PauliWordTrait};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use std::collections::hash_map::Entry;

const NONE: usize = usize::MAX;
type Word<const N: usize> = PauliWord<[u8; N]>;
type Sum<const N: usize> = FxHashMap<Word<N>, Term>;
type Evaluation = (f64, Vec<f64>, usize);

#[derive(Clone, Copy)]
struct Term {
    value: f64,
    id: usize,
    weight: f64,
}

impl Term {
    fn constant(value: f64) -> Self {
        Self {
            value,
            id: NONE,
            weight: 1.,
        }
    }

    fn sign(mut self, negative: bool) -> Self {
        if negative {
            self.value = -self.value;
            self.weight = -self.weight;
        }
        self
    }
}

struct Tape {
    edges: Vec<[(usize, f64); 2]>,
    enabled: bool,
    limit: usize,
}

impl Tape {
    fn new(parameters: usize, enabled: bool, limit: usize) -> PyResult<Self> {
        if enabled && parameters > limit {
            return Err(PyMemoryError::new_err("Pauli gradient exceeds max_tape."));
        }
        Ok(Self {
            edges: vec![[(NONE, 0.); 2]; if enabled { parameters } else { 0 }],
            enabled,
            limit,
        })
    }

    fn node(&mut self, value: f64, edges: [(usize, f64); 2]) -> PyResult<Term> {
        if !self.enabled || edges.iter().all(|&(id, weight)| id == NONE || weight == 0.) {
            return Ok(Term::constant(value));
        }
        if self.edges.len() == self.limit {
            return Err(PyMemoryError::new_err("Pauli gradient exceeds max_tape."));
        }
        let id = self.edges.len();
        self.edges.push(edges);
        Ok(Term {
            value,
            id,
            weight: 1.,
        })
    }

    fn add(&mut self, a: Term, b: Term) -> PyResult<Term> {
        self.node(a.value + b.value, [(a.id, a.weight), (b.id, b.weight)])
    }

    fn rotate(
        &mut self,
        a: Term,
        parameter: usize,
        factor: f64,
        derivative: f64,
    ) -> PyResult<Term> {
        self.node(
            a.value * factor,
            [(a.id, a.weight * factor), (parameter, a.value * derivative)],
        )
    }
}

fn insert<const N: usize>(
    sum: &mut Sum<N>,
    word: Word<N>,
    term: Term,
    tape: &mut Tape,
    limit: usize,
) -> PyResult<()> {
    if term.value == 0. && term.id == NONE {
        return Ok(());
    }
    let size = sum.len();
    match sum.entry(word) {
        Entry::Occupied(mut entry) => {
            let value = tape.add(*entry.get(), term)?;
            if value.value == 0. && value.id == NONE {
                entry.remove();
            } else {
                entry.insert(value);
            }
        }
        Entry::Vacant(entry) => {
            if size == limit {
                return Err(PyMemoryError::new_err(
                    "Pauli expansion exceeds max_terms; no terms were truncated.",
                ));
            }
            entry.insert(term);
        }
    }
    Ok(())
}

#[pyclass]
pub struct PauliCircuit {
    n: usize,
    n_inputs: usize,
    outputs: Vec<usize>,
    operations: Vec<(u8, usize, usize)>,
    parameters: usize,
    max_terms: usize,
    max_tape: usize,
}

#[pymethods]
impl PauliCircuit {
    #[new]
    #[pyo3(signature = (n, n_inputs, outputs, operations, parameters, max_terms=100000, max_tape=5000000))]
    fn new(
        n: usize,
        n_inputs: usize,
        outputs: Vec<usize>,
        operations: Vec<(u8, usize, usize)>,
        parameters: usize,
        max_terms: usize,
        max_tape: usize,
    ) -> PyResult<Self> {
        if n > 1024 || n_inputs > n || max_terms == 0 || max_tape == 0 {
            return Err(PyValueError::new_err(
                "Pauli circuits require at most 1024 wires and positive memory limits.",
            ));
        }
        let mut ordered = outputs.clone();
        ordered.sort_unstable();
        if ordered != (0..n).collect::<Vec<_>>() {
            return Err(PyValueError::new_err(
                "Output wires must be a permutation of the logical wires.",
            ));
        }
        if operations.iter().any(|&(kind, wire, operand)| {
            wire >= n
                || match kind {
                    0 => operand >= parameters,
                    1 => operand >= n || operand == wire,
                    _ => true,
                }
        }) {
            return Err(PyValueError::new_err(
                "Invalid Pauli propagation operation.",
            ));
        }
        Ok(Self {
            n,
            n_inputs,
            outputs,
            operations,
            parameters,
            max_terms,
            max_tape,
        })
    }

    #[pyo3(signature = (angles, terms, constant=0., gradient=false, input_re=None, input_im=None))]
    #[allow(clippy::too_many_arguments)]
    fn expectation(
        &self,
        py: Python<'_>,
        angles: Vec<f64>,
        terms: Vec<(String, f64)>,
        constant: f64,
        gradient: bool,
        input_re: Option<Vec<f64>>,
        input_im: Option<Vec<f64>>,
    ) -> PyResult<Evaluation> {
        if angles.len() != self.parameters
            || angles.iter().any(|x| !x.is_finite())
            || !constant.is_finite()
        {
            return Err(PyValueError::new_err(
                "Expected one finite angle per measurement.",
            ));
        }
        if terms.iter().any(|(word, c)| {
            word.len() != self.n || !word.bytes().all(|c| b"IXYZ".contains(&c)) || !c.is_finite()
        }) {
            return Err(PyValueError::new_err(
                "Expected finite Pauli coefficients and one I/X/Y/Z per output.",
            ));
        }
        let input = match (input_re, input_im) {
            (None, None) => None,
            (Some(re), Some(im)) => {
                if self.n_inputs >= usize::BITS as usize
                    || re.len() != (1usize << self.n_inputs)
                    || im.len() != re.len()
                {
                    return Err(PyValueError::new_err("Input vector has the wrong size."));
                }
                let state: Vec<_> = re
                    .into_iter()
                    .zip(im)
                    .map(|(re, im)| C::new(re, im))
                    .collect();
                let norm: f64 = state.iter().map(C::norm_sqr).sum();
                if !norm.is_finite() || (norm - 1.).abs() > 1e-5 {
                    return Err(PyValueError::new_err(
                        "Input vector must be normalized and finite.",
                    ));
                }
                Some(state)
            }
            _ => return Err(PyValueError::new_err("Provide both input_re and input_im.")),
        };
        py.allow_threads(|| match self.n {
            0..=8 => self.evaluate::<1>(&angles, &terms, constant, gradient, input.as_deref()),
            9..=32 => self.evaluate::<4>(&angles, &terms, constant, gradient, input.as_deref()),
            33..=128 => self.evaluate::<16>(&angles, &terms, constant, gradient, input.as_deref()),
            129..=512 => self.evaluate::<64>(&angles, &terms, constant, gradient, input.as_deref()),
            _ => self.evaluate::<128>(&angles, &terms, constant, gradient, input.as_deref()),
        })
    }
}

impl PauliCircuit {
    fn input_expectation<const N: usize>(&self, word: &Word<N>, input: Option<&[C]>) -> f64 {
        let Some(input) = input else {
            return if (0..self.n).any(|q| word.get_zbit(q)) {
                0.
            } else {
                1.
            };
        };
        if (self.n_inputs..self.n).any(|q| word.get_zbit(q)) {
            return 0.;
        }
        let (mut x, mut z, mut ys) = (0usize, 0usize, 0u32);
        for q in 0..self.n_inputs {
            let xb = usize::from(word.get_xbit(q));
            let zb = usize::from(word.get_zbit(q));
            x = (x << 1) | xb;
            z = (z << 1) | zb;
            ys += (xb & zb) as u32;
        }
        let phase = [
            C::new(1., 0.),
            C::new(0., 1.),
            C::new(-1., 0.),
            C::new(0., -1.),
        ][(ys % 4) as usize];
        input
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i & z).count_ones() % 2 == 0 {
                    1.
                } else {
                    -1.
                };
                (input[i ^ x].conj() * phase * sign * amp).re
            })
            .sum()
    }

    fn evaluate<const N: usize>(
        &self,
        angles: &[f64],
        terms: &[(String, f64)],
        constant: f64,
        gradient: bool,
        input: Option<&[C]>,
    ) -> PyResult<Evaluation> {
        let mut tape = Tape::new(self.parameters, gradient, self.max_tape)?;
        let mut sum = Sum::<N>::default();
        insert(
            &mut sum,
            Word::new(self.n),
            Term::constant(constant),
            &mut tape,
            self.max_terms,
        )?;
        for (text, coeff) in terms {
            let mut word = Word::new(self.n);
            for (ch, &wire) in text.bytes().zip(&self.outputs) {
                word.set_xbit(wire, ch == b'X' || ch == b'Y');
                word.set_zbit(wire, ch == b'Z' || ch == b'Y');
            }
            word.rehash();
            insert(
                &mut sum,
                word,
                Term::constant(*coeff),
                &mut tape,
                self.max_terms,
            )?;
        }
        let mut peak = sum.len();
        let mut scratch = Sum::default();
        for &(kind, wire, operand) in self.operations.iter().rev() {
            let (sin, cos) = if kind == 0 {
                super::tensor::sin_cos(angles[operand])
            } else {
                (0., 1.)
            };
            for (word, term) in sum.drain() {
                let mut phased: PhasedPauliWord<[u8; N]> = word.into();
                if kind == 1 {
                    phased.cz(wire, operand);
                } else {
                    phased.h(wire);
                }
                let mut word = phased.word;
                let term = term.sign(!phased.is_positive());
                if kind == 0 && word.get_xbit(wire) {
                    let same = tape.rotate(term, operand, cos, -sin)?;
                    insert(&mut scratch, word, same, &mut tape, self.max_terms)?;
                    let sign = if word.get_zbit(wire) { -1. } else { 1. };
                    word.set_zbit(wire, !word.get_zbit(wire));
                    word.rehash();
                    let other = tape.rotate(term, operand, sign * sin, sign * cos)?;
                    insert(&mut scratch, word, other, &mut tape, self.max_terms)?;
                } else {
                    insert(&mut scratch, word, term, &mut tape, self.max_terms)?;
                }
            }
            std::mem::swap(&mut sum, &mut scratch);
            peak = peak.max(sum.len());
        }
        let mut value = 0.;
        let mut adjoints = vec![0.; tape.edges.len()];
        for (word, term) in sum {
            let mean = self.input_expectation(&word, input);
            value += term.value * mean;
            if term.id != NONE {
                adjoints[term.id] += mean * term.weight;
            }
        }
        for i in (self.parameters..tape.edges.len()).rev() {
            let adjoint = adjoints[i];
            for &(parent, weight) in &tape.edges[i] {
                if parent != NONE {
                    adjoints[parent] += adjoint * weight;
                }
            }
        }
        adjoints.truncate(if gradient { self.parameters } else { 0 });
        if !value.is_finite() || adjoints.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "Pauli propagation produced non-finite coefficients.",
            ));
        }
        Ok((value, adjoints, peak))
    }
}
