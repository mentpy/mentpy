//! Pauli Lie closure with packed symplectic words and a linear-size work queue.

use fxhash::FxHashSet;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::VecDeque;

struct Pairs {
    left: usize,
    next: usize,
    end: usize,
}

fn closure(data: &[u8], rows: usize, qubits: usize, max_iter: usize) -> PyResult<(Vec<u8>, usize)> {
    let columns = qubits
        .checked_mul(2)
        .ok_or_else(|| PyValueError::new_err("Tableau dimensions overflow."))?;
    if rows.checked_mul(columns) != Some(data.len()) || data.iter().any(|&v| v > 1) {
        return Err(PyValueError::new_err("Expected a binary Pauli tableau."));
    }
    let words = qubits.div_ceil(64);
    let mut basis = Vec::new();
    let mut seen = FxHashSet::default();
    for row in 0..rows {
        let mut packed = vec![0u64; 2 * words];
        for q in 0..qubits {
            packed[q / 64] |= (data[row * columns + q] as u64) << (q % 64);
            packed[words + q / 64] |= (data[row * columns + qubits + q] as u64) << (q % 64);
        }
        if seen.insert(packed.clone()) {
            basis.push(packed);
        }
    }

    // Store ranges of pairs rather than materializing O(basis.len()^2) pairs.
    // This retains breadth-first discovery and the original generator order.
    let mut queue: VecDeque<_> = (0..basis.len().saturating_sub(1))
        .map(|left| Pairs {
            left,
            next: left + 1,
            end: basis.len(),
        })
        .collect();
    let mut iterations = 0;
    while let Some(pairs) = queue.front_mut() {
        if iterations == max_iter {
            return Err(PyValueError::new_err("Max iterations reached"));
        }
        iterations += 1;
        let (left, right) = (pairs.left, pairs.next);
        pairs.next += 1;
        if pairs.next == pairs.end {
            queue.pop_front();
        }
        let (left, right) = (&basis[left], &basis[right]);
        let parity = (0..words).fold(0, |parity, i| {
            parity ^ ((left[i] & right[words + i]) ^ (left[words + i] & right[i])).count_ones()
        });
        if parity & 1 == 0 {
            continue;
        }
        let product: Vec<u64> = left.iter().zip(right).map(|(x, y)| x ^ y).collect();
        if seen.insert(product.clone()) {
            let index = basis.len();
            basis.push(product);
            queue.push_back(Pairs {
                left: index,
                next: 0,
                end: index,
            });
        }
    }

    // Preserve MentPy's convention of including the identity in the result.
    let identity = vec![0u64; 2 * words];
    if seen.insert(identity.clone()) {
        basis.push(identity);
    }
    let size = basis
        .len()
        .checked_mul(columns)
        .ok_or_else(|| PyValueError::new_err("Completed tableau dimensions overflow."))?;
    let mut result = vec![0u8; size];
    for (row, packed) in basis.iter().enumerate() {
        for q in 0..qubits {
            result[row * columns + q] = ((packed[q / 64] >> (q % 64)) & 1) as u8;
            result[row * columns + qubits + q] = ((packed[words + q / 64] >> (q % 64)) & 1) as u8;
        }
    }
    Ok((result, basis.len()))
}

#[pyfunction]
pub fn lie_closure(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    rows: usize,
    qubits: usize,
    max_iter: usize,
) -> PyResult<(Py<PyBytes>, usize)> {
    let data = data.as_bytes();
    let (result, rows) = py.allow_threads(|| closure(data, rows, qubits, max_iter))?;
    Ok((PyBytes::new(py, &result).unbind(), rows))
}
