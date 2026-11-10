//! Local Clifford equivalence of graph states, arXiv:quant-ph/0405023.

use crate::{get_bit, gf2_rref_packed_impl, set_bit, word_count, PackedMatrix};
use petgraph::unionfind::UnionFind;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn equivalent(left: &[u8], right: &[u8], n: usize) -> PyResult<Option<Vec<u8>>> {
    let size = n
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("Graph dimensions overflow."))?;
    let columns = n
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("Graph dimensions overflow."))?;
    if left.len() != size || right.len() != size {
        return Err(PyValueError::new_err("Expected square adjacency matrices."));
    }
    for graph in [left, right] {
        if graph.iter().any(|&x| x > 1)
            || (0..n).any(|i| {
                graph[i * n + i] != 0 || (0..i).any(|j| graph[i * n + j] != graph[j * n + i])
            })
        {
            return Err(PyValueError::new_err("Expected simple undirected graphs."));
        }
    }
    if n == 0 {
        return Ok(None);
    }
    // Local complementation preserves connected components. The pair-of-basis
    // shortcut below applies within a connected component, not to independent
    // tensor factors (even two isolated vertices would otherwise fail).
    let components = |graph: &[u8]| {
        let mut sets = UnionFind::new(n);
        for i in 0..n {
            for j in 0..i {
                if graph[i * n + j] != 0 {
                    sets.union(i, j);
                }
            }
        }
        sets
    };
    let left_sets = components(left);
    let right_sets = components(right);
    let mut groups = vec![Vec::new(); n];
    for i in 0..n {
        groups[left_sets.find(i)].push(i);
    }
    let mut used = vec![false; n];
    let mut result = vec![0; columns];
    for group in groups.iter().filter(|group| !group.is_empty()) {
        let root = right_sets.find(group[0]);
        if used[root] || group.iter().any(|&i| right_sets.find(i) != root) {
            return Ok(None);
        }
        used[root] = true;
        if group.len() == n {
            return Ok(connected(left, right, n));
        }
        let submatrix = |graph: &[u8]| {
            group
                .iter()
                .flat_map(|&i| group.iter().map(move |&j| graph[i * n + j]))
                .collect::<Vec<_>>()
        };
        let Some(solution) = connected(&submatrix(left), &submatrix(right), group.len()) else {
            return Ok(None);
        };
        for (index, &vertex) in group.iter().enumerate() {
            for block in 0..4 {
                result[block * n + vertex] = solution[block * group.len() + index];
            }
        }
    }
    Ok(Some(result))
}

fn connected(left: &[u8], right: &[u8], n: usize) -> Option<Vec<u8>> {
    let columns = 4 * n;
    let size = n * n;
    let words = word_count(columns);
    let mut rows = vec![vec![0u64; words]; size];
    for j in 0..n {
        for k in 0..n {
            let row = &mut rows[j * n + k];
            // Equation (4): G[j,k] a[k] + delta[j,k] b[j]
            //              + sum_i G[i,j] H[i,k] c[i] + H[j,k] d[j] = 0.
            if left[j * n + k] != 0 {
                set_bit(row, k);
            }
            if j == k {
                set_bit(row, n + j);
            }
            for i in 0..n {
                if left[i * n + j] & right[i * n + k] != 0 {
                    set_bit(row, 2 * n + i);
                }
            }
            if right[j * n + k] != 0 {
                set_bit(row, 3 * n + j);
            }
        }
    }
    let (reduced, pivots) = gf2_rref_packed_impl(PackedMatrix { rows }, size, columns);
    let mut is_pivot = vec![false; columns];
    for &col in &pivots {
        is_pivot[col] = true;
    }
    let mut basis = Vec::new();
    for (free, &pivot) in is_pivot.iter().enumerate() {
        if pivot {
            continue;
        }
        let mut vector = vec![0u64; words];
        set_bit(&mut vector, free);
        for (row, &col) in pivots.iter().enumerate() {
            if get_bit(&reduced.rows[row], free) != 0 {
                set_bit(&mut vector, col);
            }
        }
        basis.push(vector);
    }
    let valid = |vector: &[u64]| {
        (0..n).all(|i| {
            (get_bit(vector, i) & get_bit(vector, 3 * n + i))
                ^ (get_bit(vector, n + i) & get_bit(vector, 2 * n + i))
                == 1
        })
    };
    let unpack = |vector: &[u64]| (0..columns).map(|i| get_bit(vector, i) as u8).collect();
    let mut candidate = vec![0u64; words];
    if basis.len() <= 4 {
        for mask in 0..1usize << basis.len() {
            candidate.fill(0);
            for (i, vector) in basis.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    for (target, value) in candidate.iter_mut().zip(vector) {
                        *target ^= value;
                    }
                }
            }
            if valid(&candidate) {
                return Some(unpack(&candidate));
            }
        }
    } else {
        // Lemma 1: above dimension four it suffices to check sums of two
        // basis vectors, giving a polynomial search rather than 2^dimension.
        for (i, first) in basis.iter().enumerate() {
            for second in &basis[i + 1..] {
                for ((target, left), right) in candidate.iter_mut().zip(first).zip(second) {
                    *target = left ^ right;
                }
                if valid(&candidate) {
                    return Some(unpack(&candidate));
                }
            }
        }
    }
    None
}

#[pyfunction]
pub fn lc_equivalent(
    py: Python<'_>,
    left: &Bound<'_, PyBytes>,
    right: &Bound<'_, PyBytes>,
    qubits: usize,
) -> PyResult<Option<Py<PyBytes>>> {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    let result = py.allow_threads(|| equivalent(left, right, qubits))?;
    Ok(result.map(|data| PyBytes::new(py, &data).unbind()))
}
