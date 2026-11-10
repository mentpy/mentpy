// Copyright 2026 Luis Mantilla
// Licensed under the Apache License, Version 2.0.
// See <http://www.apache.org/licenses/LICENSE-2.0> for details.
//! Packed GF(2) elimination and complete generalized/Pauli-flow solves.

use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::DiGraph;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::BTreeSet;

#[derive(Clone)]
struct Bits {
    rows: Vec<Vec<u64>>,
    cols: usize,
}

impl Bits {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows: vec![vec![0; cols.div_ceil(64)]; rows],
            cols,
        }
    }

    fn from_bytes(data: &[u8], rows: usize, cols: usize) -> PyResult<Self> {
        if rows
            .checked_add(cols)
            .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(PyValueError::new_err("Matrix dimensions overflow."));
        }
        let size = rows
            .checked_mul(cols)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| PyValueError::new_err("Matrix dimensions overflow."))?;
        if data.len() != size {
            return Err(PyValueError::new_err(
                "Byte input length does not match matrix dimensions.",
            ));
        }
        let mut out = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if data[i * cols + j] & 1 != 0 {
                    out.set(i, j);
                }
            }
        }
        Ok(out)
    }

    fn bytes(&self) -> Vec<u8> {
        (0..self.rows.len())
            .flat_map(|i| (0..self.cols).map(move |j| u8::from(self.get(i, j))))
            .collect()
    }

    fn get(&self, row: usize, col: usize) -> bool {
        self.rows[row][col / 64] & (1 << (col % 64)) != 0
    }

    fn set(&mut self, row: usize, col: usize) {
        self.rows[row][col / 64] |= 1 << (col % 64);
    }

    fn xor(&mut self, target: usize, source: usize) {
        if target == source {
            self.rows[target].fill(0);
            return;
        }
        if target < source {
            let (a, b) = self.rows.split_at_mut(source);
            for (x, y) in a[target].iter_mut().zip(&b[0]) {
                *x ^= y;
            }
        } else {
            let (a, b) = self.rows.split_at_mut(target);
            for (x, y) in b[0].iter_mut().zip(&a[source]) {
                *x ^= y;
            }
        }
    }

    fn leading(&self, row: usize, cols: usize) -> Option<usize> {
        for (word, value) in self.rows[row].iter().enumerate() {
            if *value != 0 {
                let col = word * 64 + value.trailing_zeros() as usize;
                return (col < cols).then_some(col);
            }
        }
        None
    }

    fn echelon(&mut self, cols: usize, reduced: bool) -> Vec<usize> {
        let mut pivots = Vec::new();
        for col in 0..cols {
            let next = pivots.len();
            let Some(pivot) = (next..self.rows.len()).find(|&i| self.get(i, col)) else {
                continue;
            };
            self.rows.swap(next, pivot);
            let start = if reduced { 0 } else { next + 1 };
            for row in start..self.rows.len() {
                if row != next && self.get(row, col) {
                    self.xor(row, next);
                }
            }
            pivots.push(col);
            if pivots.len() == self.rows.len() {
                break;
            }
        }
        pivots
    }

    fn augmented(&self) -> Self {
        let n = self.rows.len();
        let mut out = Self::zeros(n, self.cols + n);
        for i in 0..n {
            for j in 0..self.cols {
                if self.get(i, j) {
                    out.set(i, j);
                }
            }
            out.set(i, self.cols + i);
        }
        out
    }

    fn matmul(&self, other: &Self) -> Self {
        let mut out = Self::zeros(self.rows.len(), other.cols);
        for i in 0..self.rows.len() {
            for (block, &word) in self.rows[i].iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let k = block * 64 + bits.trailing_zeros() as usize;
                    for (x, y) in out.rows[i].iter_mut().zip(&other.rows[k]) {
                        *x ^= y;
                    }
                    bits &= bits - 1;
                }
            }
        }
        out
    }

    fn graph(&self) -> DiGraph<(), ()> {
        let mut graph = DiGraph::<(), ()>::with_capacity(self.cols, 0);
        let vertices: Vec<_> = (0..self.cols).map(|_| graph.add_node(())).collect();
        for row in 0..self.rows.len() {
            for col in 0..self.cols {
                if self.get(row, col) {
                    graph.add_edge(vertices[col], vertices[row], ());
                }
            }
        }
        graph
    }

    fn is_dag(&self) -> bool {
        !is_cyclic_directed(&self.graph())
    }
}

fn dag_inverse(nl: &Bits, nr: &Bits) -> Option<Bits> {
    let n = nl.cols;
    let free = nr.cols;
    if free == 0 {
        return nl.is_dag().then(|| Bits::zeros(0, n));
    }
    let mut initial = Bits::zeros(n, free + 2 * n);
    for i in 0..n {
        for j in 0..free {
            if nr.get(i, j) {
                initial.set(i, j);
            }
        }
        for j in 0..n {
            if nl.get(i, j) {
                initial.set(i, free + j);
            }
        }
        initial.set(i, free + n + i);
    }
    let mut system = initial.clone();
    system.echelon(free, false);
    let mut solved = vec![false; n];
    let mut p = Bits::zeros(free, n);
    while solved.iter().any(|&x| !x) {
        let zero_row = (0..n)
            .find(|&i| system.leading(i, free).is_none())
            .unwrap_or(n);
        let layer: Vec<_> = (0..n)
            .filter(|&v| !solved[v] && (zero_row..n).all(|i| !system.get(i, free + v)))
            .collect();
        if layer.is_empty() {
            return None;
        }
        for &vertex in &layer {
            // Solve every layer RHS using the same row-echelon form.
            for row in (0..n).rev() {
                let Some(pivot) = system.leading(row, free) else {
                    continue;
                };
                let mut value = system.get(row, free + vertex);
                for col in pivot + 1..free {
                    value ^= system.get(row, col) && p.get(col, vertex);
                }
                if value {
                    p.set(pivot, vertex);
                }
            }
        }
        for vertex in layer {
            solved[vertex] = true;
            let tracker = free + n + vertex;
            let dependent: Vec<_> = (0..n).filter(|&i| system.get(i, tracker)).collect();
            let Some(&last) = dependent.last() else {
                continue;
            };
            for &row in &dependent[..dependent.len() - 1] {
                system.xor(row, last);
            }
            for (x, y) in system.rows[last].iter_mut().zip(&initial.rows[vertex]) {
                *x ^= y;
            }
            for row in 0..n {
                if row == last {
                    continue;
                }
                let Some(pivot) = system.leading(row, free) else {
                    break;
                };
                if system.get(last, pivot) {
                    system.xor(last, row);
                }
            }
            let mut order: Vec<_> = (0..n).collect();
            order.sort_by_key(|&i| (system.leading(i, free).unwrap_or(free), i));
            system.rows = order.into_iter().map(|i| system.rows[i].clone()).collect();
        }
    }
    let mut relation = nr.matmul(&p);
    for (row, nl_row) in relation.rows.iter_mut().zip(&nl.rows) {
        for (x, y) in row.iter_mut().zip(nl_row) {
            *x ^= y;
        }
    }
    relation.is_dag().then_some(p)
}

fn pflow_impl(m: Bits, order: Bits) -> Option<(Bits, Bits)> {
    let n = m.rows.len();
    let cols = m.cols;
    if n == 0 {
        return Some((Bits::zeros(cols, 0), Bits::zeros(0, 0)));
    }
    if cols < n {
        return None;
    }
    let mut augmented = m.augmented();
    let pivots = augmented.echelon(cols, true);
    if pivots.len() != n {
        return None;
    }
    if cols == n {
        let mut c = Bits::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if augmented.get(i, n + j) {
                    c.set(i, j);
                }
            }
        }
        let relation = order.matmul(&c);
        return relation.is_dag().then_some((c, relation));
    }
    let mut basis = Bits::zeros(cols, cols);
    for (i, &pivot) in pivots.iter().enumerate() {
        for j in 0..n {
            if augmented.get(i, cols + j) {
                basis.set(pivot, j);
            }
        }
    }
    let mut free_index = n;
    for free_col in 0..cols {
        if pivots.contains(&free_col) {
            continue;
        }
        basis.set(free_col, free_index);
        for (i, &pivot) in pivots.iter().enumerate() {
            if augmented.get(i, free_col) {
                basis.set(pivot, free_index);
            }
        }
        free_index += 1;
    }
    let changed = order.matmul(&basis);
    let mut nl = Bits::zeros(n, n);
    let mut nr = Bits::zeros(n, cols - n);
    for i in 0..n {
        for j in 0..n {
            if changed.get(i, j) {
                nl.set(i, j);
            }
        }
        for j in n..cols {
            if changed.get(i, j) {
                nr.set(i, j - n);
            }
        }
    }
    let p = dag_inverse(&nl, &nr)?;
    let mut changed_c = Bits::zeros(cols, n);
    for i in 0..n {
        changed_c.set(i, i);
    }
    for i in n..cols {
        changed_c.rows[i] = p.rows[i - n].clone();
    }
    let c = basis.matmul(&changed_c);
    let relation = order.matmul(&c);
    relation.is_dag().then_some((c, relation))
}

type BytePair = (Py<PyBytes>, Py<PyBytes>);

#[pyfunction]
pub fn flow_layers(py: Python<'_>, relation: &[u8], n: usize) -> PyResult<Vec<usize>> {
    let relation = Bits::from_bytes(relation, n, n)?;
    py.allow_threads(|| {
        let graph = relation.graph();
        let order = toposort(&graph, None)
            .map_err(|_| PyValueError::new_err("The induced Pauli-flow relation is cyclic."))?;
        let mut layers = vec![1; n];
        for node in order.into_iter().rev() {
            layers[node.index()] = 1 + graph
                .neighbors(node)
                .map(|next| layers[next.index()])
                .max()
                .unwrap_or(0);
        }
        Ok(layers)
    })
}

#[pyfunction]
pub fn pflow_matrices(
    py: Python<'_>,
    m: &[u8],
    order: &[u8],
    rows: usize,
    cols: usize,
) -> PyResult<Option<BytePair>> {
    let m = Bits::from_bytes(m, rows, cols)?;
    let order = Bits::from_bytes(order, rows, cols)?;
    let result = py.allow_threads(|| pflow_impl(m, order));
    Ok(result.map(|(c, r)| {
        (
            PyBytes::new(py, &c.bytes()).unbind(),
            PyBytes::new(py, &r.bytes()).unbind(),
        )
    }))
}

#[pyfunction]
pub fn solve_dag(
    py: Python<'_>,
    nl: &[u8],
    nr: &[u8],
    n: usize,
    free: usize,
) -> PyResult<Option<Py<PyBytes>>> {
    let nl = Bits::from_bytes(nl, n, n)?;
    let nr = Bits::from_bytes(nr, n, free)?;
    Ok(py
        .allow_threads(|| dag_inverse(&nl, &nr))
        .map(|p| PyBytes::new(py, &p.bytes()).unbind()))
}

fn gflow_impl(
    adjacency: Bits,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
) -> Option<(Vec<u8>, Vec<usize>)> {
    let n = adjacency.cols;
    let mut input = vec![false; n];
    let mut done = vec![false; n];
    for i in inputs {
        input[i] = true;
    }
    for i in outputs {
        done[i] = true;
    }
    let mut layers = vec![0; n];
    let mut corrections = Bits::zeros(n, n);
    let mut depth = 1;
    while done.iter().any(|&x| !x) {
        let rows: Vec<_> = (0..n).filter(|&i| !done[i]).collect();
        let cols: Vec<_> = (0..n).filter(|&i| done[i] && !input[i]).collect();
        let mut system = Bits::zeros(rows.len(), cols.len() + rows.len());
        for (i, &row) in rows.iter().enumerate() {
            for (j, &col) in cols.iter().enumerate() {
                if adjacency.get(row, col) {
                    system.set(i, j);
                }
            }
            system.set(i, cols.len() + i);
        }
        let pivots = system.echelon(cols.len(), true);
        let mut found = false;
        for (i, &vertex) in rows.iter().enumerate() {
            if (pivots.len()..rows.len()).any(|row| system.get(row, cols.len() + i)) {
                continue;
            }
            for (row, &pivot) in pivots.iter().enumerate() {
                if system.get(row, cols.len() + i) {
                    corrections.set(vertex, cols[pivot]);
                }
            }
            layers[vertex] = depth;
            done[vertex] = true;
            found = true;
        }
        if !found {
            return None;
        }
        depth += 1;
    }
    Some((corrections.bytes(), layers))
}

#[pyfunction]
pub fn gflow(
    py: Python<'_>,
    adjacency: &[u8],
    n: usize,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
) -> PyResult<Option<(Py<PyBytes>, Vec<usize>)>> {
    let adjacency = Bits::from_bytes(adjacency, n, n)?;
    if inputs.iter().chain(&outputs).any(|&i| i >= n) {
        return Err(PyValueError::new_err(
            "Input and output indices must be graph vertices.",
        ));
    }
    Ok(py
        .allow_threads(|| gflow_impl(adjacency, inputs, outputs))
        .map(|(c, layers)| (PyBytes::new(py, &c).unbind(), layers)))
}

#[pyfunction]
pub fn cflow(
    py: Python<'_>,
    adjacency: Vec<Vec<usize>>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
) -> PyResult<Option<(Vec<usize>, Vec<usize>)>> {
    let n = adjacency.len();
    if adjacency
        .iter()
        .flatten()
        .chain(&inputs)
        .chain(&outputs)
        .any(|&i| i >= n)
    {
        return Err(PyValueError::new_err(
            "Input, output and neighbor indices must be graph vertices.",
        ));
    }
    Ok(py.allow_threads(|| {
        let mut input = vec![false; n];
        let mut done = vec![false; n];
        for i in inputs {
            input[i] = true;
        }
        for i in outputs {
            done[i] = true;
        }
        let mut past = vec![0; n];
        let mut candidates = BTreeSet::new();
        for v in 0..n {
            if done[v] && !input[v] {
                past[v] = adjacency[v].iter().filter(|&&u| !done[u]).count();
                if past[v] == 1 {
                    candidates.insert(v);
                }
            }
        }
        let mut layers = vec![0; n];
        let mut corrections = vec![n; n];
        let mut depth = 1;
        while !candidates.is_empty() {
            let mut next = BTreeSet::new();
            for v in candidates {
                let remaining: Vec<_> =
                    adjacency[v].iter().copied().filter(|&u| !done[u]).collect();
                if remaining.len() != 1 {
                    continue;
                }
                let u = remaining[0];
                corrections[u] = v;
                layers[u] = depth;
                done[u] = true;
                if !input[u] {
                    past[u] = adjacency[u].iter().filter(|&&w| !done[w]).count();
                    if past[u] == 1 {
                        next.insert(u);
                    }
                }
                for &w in &adjacency[u] {
                    if past[w] > 0 {
                        past[w] -= 1;
                        if past[w] == 1 {
                            next.insert(w);
                        }
                    }
                }
            }
            candidates = next;
            depth += 1;
        }
        done.iter().all(|&x| x).then_some((corrections, layers))
    }))
}

#[pyfunction]
pub fn gf2_solve(
    py: Python<'_>,
    data: &[u8],
    rhs: &[u8],
    rows: usize,
    cols: usize,
) -> PyResult<Option<Py<PyBytes>>> {
    let matrix = Bits::from_bytes(data, rows, cols)?;
    if rhs.len() != rows {
        return Err(PyValueError::new_err(
            "Right-hand side length does not match the matrix.",
        ));
    }
    let rhs = rhs.to_vec();
    let result = py.allow_threads(|| {
        let mut system = Bits::zeros(rows, cols + 1);
        for (i, &value) in rhs.iter().enumerate() {
            for j in 0..cols {
                if matrix.get(i, j) {
                    system.set(i, j);
                }
            }
            if value & 1 != 0 {
                system.set(i, cols);
            }
        }
        let pivots = system.echelon(cols, true);
        if (pivots.len()..rows).any(|i| system.get(i, cols)) {
            return None;
        }
        let mut solution = vec![0; cols];
        for (row, &pivot) in pivots.iter().enumerate() {
            solution[pivot] = u8::from(system.get(row, cols));
        }
        Some(solution)
    });
    Ok(result.map(|x| PyBytes::new(py, &x).unbind()))
}
