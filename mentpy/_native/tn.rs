// Copyright 2026 Luis Mantilla
// Licensed under the Apache License, Version 2.0.
// See <http://www.apache.org/licenses/LICENSE-2.0> for details.
//! Graph-state tensor networks with Candle reverse-mode differentiation.

use super::contract::{Complex, Plan};
use super::tensor;
use candle_core::{Device, IndexOp, Result, Tensor, Var};
use num_complex::Complex64 as C;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeSet;

enum Operand {
    Constant(Complex),
    Input,
    Measurement(usize),
}

fn error(error: candle_core::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn device(name: &str) -> Result<Device> {
    if name == "cpu" {
        return Ok(Device::Cpu);
    }
    if name == "cuda" || name.starts_with("cuda:") {
        let index = name
            .strip_prefix("cuda:")
            .unwrap_or("0")
            .parse::<usize>()
            .map_err(|_| candle_core::Error::Msg("Invalid CUDA device index.".into()))?;
        return Device::new_cuda(index);
    }
    candle_core::bail!("device must be 'cpu', 'cuda', or 'cuda:N'.")
}

#[pyclass]
pub struct TensorNetwork {
    operands: Vec<Operand>,
    plan: Plan,
    device: Device,
    n_inputs: usize,
    n_outputs: usize,
    n_measured: usize,
    explicit_input: bool,
    cpu_constants: Vec<Option<Vec<C>>>,
}

#[pymethods]
impl TensorNetwork {
    #[new]
    #[allow(clippy::too_many_arguments)] // Python constructor keywords.
    #[pyo3(signature = (n, edges, inputs, outputs, measured, explicit_input=false, max_elements=67108864, device_name="cpu"))]
    fn new(
        n: usize,
        edges: Vec<(usize, usize)>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        measured: Vec<usize>,
        explicit_input: bool,
        max_elements: usize,
        device_name: &str,
    ) -> PyResult<Self> {
        if inputs
            .iter()
            .chain(&outputs)
            .chain(&measured)
            .any(|&v| v >= n)
            || edges.iter().any(|&(u, v)| u >= n || v >= n || u == v)
        {
            return Err(PyValueError::new_err("Invalid tensor-network node index."));
        }
        let unique = |values: &[usize]| values.iter().copied().collect::<BTreeSet<_>>();
        let measured_set = unique(&measured);
        let output_set = unique(&outputs);
        if unique(&inputs).len() != inputs.len()
            || measured_set.len() != measured.len()
            || output_set.len() != outputs.len()
            || !measured_set.is_disjoint(&output_set)
            || measured.len() + outputs.len() != n
            || edges
                .iter()
                .map(|&(u, v)| (u.min(v), u.max(v)))
                .collect::<BTreeSet<_>>()
                .len()
                != edges.len()
        {
            return Err(PyValueError::new_err("Nodes must be unique; measurements and outputs must partition the graph, and edges must be distinct."));
        }
        let dev = device(device_name).map_err(error)?;
        let mut operands = Vec::new();
        let mut indices = Vec::new();
        let mut legs: Vec<_> = (0..n).map(|i| vec![i]).collect();
        let mut label = n;
        let mut add = |data: Vec<f64>, labels: Vec<usize>| -> Result<()> {
            let shape = vec![2; labels.len()];
            let re = Tensor::from_vec(data, shape.as_slice(), &dev)?;
            let im = re.zeros_like()?;
            operands.push(Operand::Constant(Complex { re, im }));
            indices.push(labels);
            Ok(())
        };
        for (u, v) in edges {
            legs[u].push(label);
            legs[v].push(label + 1);
            add(vec![1., 1., 1., -1.], vec![label, label + 1]).map_err(error)?;
            label += 2;
        }
        for node_legs in &legs {
            if node_legs.len() == 1 {
                add(vec![1., 1.], node_legs.clone()).map_err(error)?;
            } else {
                let mut head = node_legs[0];
                for &leg in &node_legs[1..node_legs.len() - 1] {
                    add(vec![1., 0., 0., 0., 0., 0., 0., 1.], vec![head, leg, label])
                        .map_err(error)?;
                    head = label;
                    label += 1;
                }
                add(vec![1., 0., 0., 1.], vec![head, *node_legs.last().unwrap()]).map_err(error)?;
            }
        }
        if n == 0 {
            add(vec![1.], vec![]).map_err(error)?;
        }
        if explicit_input {
            operands.push(Operand::Input);
            indices.push(inputs.clone());
        }
        for (i, &node) in measured.iter().enumerate() {
            operands.push(Operand::Measurement(i));
            indices.push(vec![node]);
        }
        let plan = Plan::new(&indices, &outputs, max_elements).map_err(error)?;
        let cpu_constants = operands
            .iter()
            .map(|operand| -> Result<_> {
                match operand {
                    Operand::Constant(value) if dev.is_cpu() => {
                        let re = value.re.flatten_all()?.to_vec1::<f64>()?;
                        let im = value.im.flatten_all()?.to_vec1::<f64>()?;
                        Ok(Some(
                            re.into_iter().zip(im).map(|(a, b)| C::new(a, b)).collect(),
                        ))
                    }
                    _ => Ok(None),
                }
            })
            .collect::<Result<Vec<_>>>()
            .map_err(error)?;
        Ok(Self {
            operands,
            plan,
            device: dev,
            n_inputs: inputs.len(),
            n_outputs: outputs.len(),
            n_measured: measured.len(),
            explicit_input,
            cpu_constants,
        })
    }

    #[getter]
    fn peak_elements(&self) -> usize {
        self.plan.peak_elements
    }

    #[pyo3(signature = (angles, input_re=None, input_im=None))]
    fn run(
        &self,
        py: Python<'_>,
        angles: Vec<f64>,
        input_re: Option<Vec<f64>>,
        input_im: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.allow_threads(|| {
            self.validate(&angles, &input_re, &input_im)?;
            if self.device.is_cpu() {
                let (state, _) = self.cpu_forward(&angles, &input_re, &input_im, false)?;
                let (state, _) = tensor::normalize(state)?;
                return Ok(state.into_iter().map(|z| (z.re, z.im)).unzip());
            }
            let parameters = Tensor::new(angles.as_slice(), &self.device)?;
            let state = self.forward(&parameters, &angles, &input_re, &input_im)?;
            Ok((state.re.to_vec1()?, state.im.to_vec1()?))
        })
        .map_err(error)
    }

    #[pyo3(signature = (angles, terms, constant=0.0, gradient=false, input_re=None, input_im=None))]
    #[allow(clippy::too_many_arguments)] // Python evaluation keywords.
    fn expectation(
        &self,
        py: Python<'_>,
        angles: Vec<f64>,
        terms: Vec<(String, f64)>,
        constant: f64,
        gradient: bool,
        input_re: Option<Vec<f64>>,
        input_im: Option<Vec<f64>>,
    ) -> PyResult<(f64, Vec<f64>)> {
        // Candle's backward traversal is recursive. Give long MBQC patterns a
        // bounded, explicit worker stack rather than relying on Python's stack.
        py.allow_threads(|| {
            if self.device.is_cpu() {
                return self
                    .evaluate(&angles, &terms, constant, gradient, &input_re, &input_im)
                    .map_err(error);
            }
            std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn_scoped(scope, || {
                        self.evaluate(&angles, &terms, constant, gradient, &input_re, &input_im)
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .join()
                    .map_err(|_| PyRuntimeError::new_err("Native tensor evaluation failed."))?
                    .map_err(error)
            })
        })
    }
}

impl TensorNetwork {
    fn cpu_forward(
        &self,
        angles: &[f64],
        re: &Option<Vec<f64>>,
        im: &Option<Vec<f64>>,
        gradient: bool,
    ) -> Result<(Vec<C>, tensor::Tape)> {
        let operands = self
            .operands
            .iter()
            .enumerate()
            .map(|(i, op)| match op {
                Operand::Constant(_) => self.cpu_constants[i].as_ref().unwrap().clone(),
                Operand::Input => re
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(im.as_ref().unwrap())
                    .map(|(&a, &b)| C::new(a, b))
                    .collect(),
                Operand::Measurement(index) => {
                    tensor::measurement(angles[2 * index], angles[2 * index + 1]).to_vec()
                }
            })
            .collect();
        tensor::contract(&self.plan, operands, gradient)
    }

    fn validate(&self, angles: &[f64], re: &Option<Vec<f64>>, im: &Option<Vec<f64>>) -> Result<()> {
        if angles.len() != 2 * self.n_measured || angles.iter().any(|x| !x.is_finite()) {
            candle_core::bail!("Expected finite azimuth and latitude angles per measured node.");
        }
        if self.explicit_input {
            let expected = 1usize
                .checked_shl(self.n_inputs as u32)
                .ok_or_else(|| candle_core::Error::Msg("Input dimension overflow.".into()))?;
            let (Some(re), Some(im)) = (re, im) else {
                candle_core::bail!("An explicit input vector is required.");
            };
            if re.len() != expected || im.len() != expected {
                candle_core::bail!("Input state has the wrong dimension.");
            }
            let norm: f64 = re.iter().zip(im).map(|(a, b)| a * a + b * b).sum();
            if !norm.is_finite() || (norm - 1.).abs() > 1e-5 {
                candle_core::bail!("Input state must be normalized.");
            }
        } else if re.is_some() || im.is_some() {
            candle_core::bail!("This network uses implicit plus-state inputs.");
        }
        Ok(())
    }

    fn forward(
        &self,
        parameters: &Tensor,
        angles: &[f64],
        input_re: &Option<Vec<f64>>,
        input_im: &Option<Vec<f64>>,
    ) -> Result<Complex> {
        let mut values = Vec::with_capacity(self.operands.len());
        let zero = Tensor::new(0f64, &self.device)?;
        for operand in &self.operands {
            let value = match operand {
                Operand::Constant(value) => value.clone(),
                Operand::Input => {
                    let shape = vec![2; self.n_inputs];
                    Complex {
                        re: Tensor::from_slice(
                            input_re.as_ref().unwrap(),
                            shape.as_slice(),
                            &self.device,
                        )?,
                        im: Tensor::from_slice(
                            input_im.as_ref().unwrap(),
                            shape.as_slice(),
                            &self.device,
                        )?,
                    }
                }
                Operand::Measurement(index) => {
                    let angle = parameters.i(2 * index)?;
                    let (mut cos, mut sin) = (angle.cos()?, angle.sin()?);
                    if let Some(quadrant) = super::tensor::clifford(angles[2 * index]) {
                        // Correct the value at exact Clifford angles while
                        // retaining derivatives of otherwise zero sine terms.
                        cos = (&cos - cos.detach())?.affine(1., [1., 0., -1., 0.][quadrant])?;
                        sin = (&sin - sin.detach())?.affine(1., [0., 1., 0., -1.][quadrant])?;
                    }
                    let polar = parameters
                        .i(2 * index + 1)?
                        .affine(-0.5, std::f64::consts::FRAC_PI_4)?;
                    let (mut first, mut second) = (polar.cos()?, polar.sin()?);
                    if let Some(quadrant) = super::tensor::clifford(
                        std::f64::consts::FRAC_PI_4 - 0.5 * angles[2 * index + 1],
                    ) {
                        first =
                            (&first - first.detach())?.affine(1., [1., 0., -1., 0.][quadrant])?;
                        second =
                            (&second - second.detach())?.affine(1., [0., 1., 0., -1.][quadrant])?;
                    }
                    Complex {
                        re: Tensor::stack(&[&(&cos * &first)?, &second], 0)?,
                        im: Tensor::stack(&[&(&sin * &first)?, &zero], 0)?,
                    }
                }
            };
            values.push(value);
        }
        let state = self.plan.contract(values)?;
        let state = Complex {
            re: state.re.flatten_all()?,
            im: state.im.flatten_all()?,
        };
        let norm = (state.re.sqr()? + state.im.sqr()?)?.sum_all()?;
        let scalar = norm.to_scalar::<f64>()?;
        if !scalar.is_finite() || scalar <= 0. {
            candle_core::bail!(
                "Selected measurement branch has zero probability or nonfinite amplitudes."
            );
        }
        let norm = norm.sqrt()?;
        Ok(Complex {
            re: state.re.broadcast_div(&norm)?,
            im: state.im.broadcast_div(&norm)?,
        })
    }

    fn evaluate(
        &self,
        angles: &[f64],
        terms: &[(String, f64)],
        constant: f64,
        gradient: bool,
        input_re: &Option<Vec<f64>>,
        input_im: &Option<Vec<f64>>,
    ) -> Result<(f64, Vec<f64>)> {
        self.validate(angles, input_re, input_im)?;
        if !constant.is_finite() || terms.iter().any(|(_, c)| !c.is_finite()) {
            candle_core::bail!("Observable coefficients must be finite.");
        }
        if self.device.is_cpu() {
            let (state, tape) = self.cpu_forward(angles, input_re, input_im, gradient)?;
            let (state, norm) = tensor::normalize(state)?;
            let (value, mut adjoint) =
                tensor::expectation(&state, terms, constant, self.n_outputs)?;
            if !gradient {
                return Ok((value, vec![]));
            }
            for z in &mut adjoint {
                *z /= norm;
            }
            let grads = tape.backward(&self.plan, adjoint);
            let mut derivative = vec![0.; angles.len()];
            for (op, grad) in self.operands.iter().zip(grads) {
                if let Operand::Measurement(i) = op {
                    let values = tensor::derivative(angles[2 * i], angles[2 * i + 1], &grad);
                    derivative[2 * i..2 * i + 2].copy_from_slice(&values);
                }
            }
            return Ok((value, derivative));
        }
        let parameters = Var::from_slice(angles, angles.len(), &self.device)?;
        let input = if gradient {
            parameters.as_tensor().clone()
        } else {
            parameters.as_detached_tensor()
        };
        let state = self.forward(&input, angles, input_re, input_im)?;
        let dimension = state.re.elem_count();
        if dimension > u32::MAX as usize {
            candle_core::bail!("Dense output exceeds supported indexing size.");
        }
        let mut cost = Tensor::new(constant, &self.device)?;
        for (word, coefficient) in terms {
            if word.len() != self.n_outputs || word.chars().any(|c| !"IXYZ".contains(c)) {
                candle_core::bail!("Observable must contain one Pauli letter per output.");
            }
            let mut flip = 0usize;
            let mut sign = 0usize;
            let mut ys = 0;
            for (i, c) in word.chars().enumerate() {
                let bit = 1usize << (self.n_outputs - i - 1);
                if c == 'X' || c == 'Y' {
                    flip |= bit;
                }
                if c == 'Z' || c == 'Y' {
                    sign |= bit;
                }
                if c == 'Y' {
                    ys += 1;
                }
            }
            let indices: Vec<_> = (0..dimension).map(|i| (i ^ flip) as u32).collect();
            let phases: Vec<f64> = (0..dimension)
                .map(|i| {
                    if (i & sign).count_ones().is_multiple_of(2) {
                        1.
                    } else {
                        -1.
                    }
                })
                .collect();
            let indices = Tensor::from_vec(indices, dimension, &self.device)?;
            let phases = Tensor::from_vec(phases, dimension, &self.device)?;
            let re = state.re.index_select(&indices, 0)?;
            let im = state.im.index_select(&indices, 0)?;
            let value = match ys % 4 {
                0 => ((&re * &state.re)? + (&im * &state.im)?)?,
                1 => ((&im * &state.re)? - (&re * &state.im)?)?,
                2 => ((&re * &state.re)? + (&im * &state.im)?)?.neg()?,
                _ => ((&re * &state.im)? - (&im * &state.re)?)?,
            };
            cost = (cost + (value * phases)?.sum_all()?.affine(*coefficient, 0.)?)?;
        }
        let value = cost.to_scalar::<f64>()?;
        let derivative = if gradient {
            let grads = cost.backward()?;
            match grads.get(parameters.as_tensor()) {
                Some(grad) => grad.to_vec1()?,
                None => vec![0.; angles.len()],
            }
        } else {
            vec![]
        };
        Ok((value, derivative))
    }
}
