use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type Matrix = Vec<Vec<u32>>;

struct PackedMatrix {
    rows: Vec<Vec<u64>>,
}

fn normalise_matrix(matrix: Matrix) -> PyResult<Matrix> {
    if matrix.is_empty() {
        return Ok(matrix);
    }

    let width = matrix[0].len();
    if matrix.iter().any(|row| row.len() != width) {
        return Err(PyValueError::new_err("GF(2) matrices must be rectangular."));
    }

    Ok(matrix
        .into_iter()
        .map(|row| row.into_iter().map(|value| value & 1).collect())
        .collect())
}

fn word_count(width: usize) -> usize {
    (width + 63) / 64
}

fn identity(size: usize) -> Matrix {
    let mut matrix = vec![vec![0; size]; size];
    for i in 0..size {
        matrix[i][i] = 1;
    }
    matrix
}

fn matrix_width(matrix: &Matrix) -> usize {
    if matrix.is_empty() {
        0
    } else {
        matrix[0].len()
    }
}

fn set_bit(row: &mut [u64], col: usize) {
    row[col / 64] |= 1u64 << (col % 64);
}

fn get_bit(row: &[u64], col: usize) -> u32 {
    ((row[col / 64] >> (col % 64)) & 1) as u32
}

fn pack_augmented(left: &Matrix, rhs: Option<&Matrix>) -> PyResult<PackedMatrix> {
    if let Some(rhs) = rhs {
        if left.len() != rhs.len() {
            return Err(PyValueError::new_err(
                "Cannot augment matrices with different row counts.",
            ));
        }
    }

    let left_width = matrix_width(left);
    let right_width = rhs.map(matrix_width).unwrap_or(0);
    let width = left_width + right_width;
    let mut rows = vec![vec![0u64; word_count(width)]; left.len()];

    for (row_index, row) in left.iter().enumerate() {
        for (col, &value) in row.iter().enumerate() {
            if value & 1 == 1 {
                set_bit(&mut rows[row_index], col);
            }
        }
    }

    if let Some(rhs) = rhs {
        for (row_index, row) in rhs.iter().enumerate() {
            for (col, &value) in row.iter().enumerate() {
                if value & 1 == 1 {
                    set_bit(&mut rows[row_index], left_width + col);
                }
            }
        }
    }

    Ok(PackedMatrix { rows })
}

fn pack_augmented_bytes(
    data: &[u8],
    row_count: usize,
    col_count: usize,
    identity_rhs: bool,
) -> PyResult<PackedMatrix> {
    let expected_len = row_count
        .checked_mul(col_count)
        .ok_or_else(|| PyValueError::new_err("Matrix dimensions overflow."))?;
    if data.len() != expected_len {
        return Err(PyValueError::new_err(
            "Byte input length does not match matrix dimensions.",
        ));
    }

    let rhs_width = if identity_rhs { row_count } else { 0 };
    let width = col_count + rhs_width;
    let mut rows = vec![vec![0u64; word_count(width)]; row_count];

    for row in 0..row_count {
        let row_offset = row * col_count;
        for col in 0..col_count {
            if data[row_offset + col] & 1 == 1 {
                set_bit(&mut rows[row], col);
            }
        }
        if identity_rhs {
            set_bit(&mut rows[row], col_count + row);
        }
    }

    Ok(PackedMatrix { rows })
}

fn xor_rows(matrix: &mut PackedMatrix, target: usize, source: usize) {
    if target < source {
        let (left, right) = matrix.rows.split_at_mut(source);
        for (target_word, source_word) in left[target].iter_mut().zip(right[0].iter()) {
            *target_word ^= source_word;
        }
    } else {
        let (left, right) = matrix.rows.split_at_mut(target);
        for (target_word, source_word) in right[0].iter_mut().zip(left[source].iter()) {
            *target_word ^= source_word;
        }
    }
}

fn gf2_rref_packed_impl(
    mut packed: PackedMatrix,
    row_count: usize,
    max_cols: usize,
) -> (PackedMatrix, Vec<usize>) {
    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0usize;

    for col in 0..max_cols {
        let pivot = (pivot_row..row_count).find(|&row| get_bit(&packed.rows[row], col) == 1);
        let Some(pivot) = pivot else {
            continue;
        };

        if pivot != pivot_row {
            packed.rows.swap(pivot, pivot_row);
        }

        for row in 0..row_count {
            if row == pivot_row || get_bit(&packed.rows[row], col) == 0 {
                continue;
            }
            xor_rows(&mut packed, row, pivot_row);
        }

        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == row_count {
            break;
        }
    }

    (packed, pivot_cols)
}

fn gf2_rref_augmented_impl(
    matrix: Matrix,
    rhs: Option<Matrix>,
    max_cols: Option<usize>,
) -> PyResult<(PackedMatrix, Vec<usize>)> {
    let matrix = normalise_matrix(matrix)?;
    let rhs = rhs.map(normalise_matrix).transpose()?;

    let row_count = matrix.len();
    let original_width = matrix_width(&matrix);
    let max_cols = max_cols.unwrap_or(original_width);
    if max_cols > original_width {
        return Err(PyValueError::new_err("max_cols exceeds matrix width."));
    }

    Ok(gf2_rref_packed_impl(
        pack_augmented(&matrix, rhs.as_ref())?,
        row_count,
        max_cols,
    ))
}

#[pyfunction]
fn gf2_inverse(matrix: Matrix) -> PyResult<Option<Matrix>> {
    let matrix = normalise_matrix(matrix)?;
    let size = matrix.len();
    if matrix.iter().any(|row| row.len() != size) {
        return Err(PyValueError::new_err(
            "Only square matrices can be inverted.",
        ));
    }

    let (augmented, pivot_cols) =
        gf2_rref_augmented_impl(matrix, Some(identity(size)), Some(size))?;
    if pivot_cols.len() != size {
        return Ok(None);
    }

    let inverse = (0..size)
        .map(|row| {
            (0..size)
                .map(|col| get_bit(&augmented.rows[row], size + col))
                .collect()
        })
        .collect();
    Ok(Some(inverse))
}

#[pyfunction]
fn gf2_inverse_bytes<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyBytes>,
    row_count: usize,
    col_count: usize,
) -> PyResult<Option<(Py<PyBytes>, usize, usize)>> {
    if row_count != col_count {
        return Err(PyValueError::new_err(
            "Only square matrices can be inverted.",
        ));
    }

    let (augmented, pivot_cols) = gf2_rref_packed_impl(
        pack_augmented_bytes(data.as_bytes(), row_count, col_count, true)?,
        row_count,
        col_count,
    );
    if pivot_cols.len() != row_count {
        return Ok(None);
    }

    let mut inverse = vec![0u8; row_count * col_count];
    for row in 0..row_count {
        for col in 0..col_count {
            inverse[row * col_count + col] = get_bit(&augmented.rows[row], col_count + col) as u8;
        }
    }

    Ok(Some((
        PyBytes::new(py, &inverse).unbind(),
        row_count,
        col_count,
    )))
}

#[pyfunction]
fn gf2_right_inverse_and_kernel(matrix: Matrix) -> PyResult<Option<(Matrix, Matrix)>> {
    let matrix = normalise_matrix(matrix)?;
    let row_count = matrix.len();
    let col_count = if row_count == 0 { 0 } else { matrix[0].len() };

    let (augmented, pivot_cols) =
        gf2_rref_augmented_impl(matrix, Some(identity(row_count)), Some(col_count))?;
    if pivot_cols.len() != row_count {
        return Ok(None);
    }

    let mut right_inverse = vec![vec![0; row_count]; col_count];
    for (row, &pivot_col) in pivot_cols.iter().enumerate() {
        for col in 0..row_count {
            right_inverse[pivot_col][col] = get_bit(&augmented.rows[row], col_count + col);
        }
    }

    let mut is_pivot = vec![false; col_count];
    for &pivot_col in &pivot_cols {
        is_pivot[pivot_col] = true;
    }
    let free_cols: Vec<usize> = (0..col_count).filter(|&col| !is_pivot[col]).collect();

    let mut kernel = vec![vec![0; free_cols.len()]; col_count];
    for (basis_col, &free_col) in free_cols.iter().enumerate() {
        kernel[free_col][basis_col] = 1;
        for (row, &pivot_col) in pivot_cols.iter().enumerate() {
            kernel[pivot_col][basis_col] = get_bit(&augmented.rows[row], free_col);
        }
    }

    Ok(Some((right_inverse, kernel)))
}

#[pyfunction]
fn gf2_right_inverse_and_kernel_bytes<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyBytes>,
    row_count: usize,
    col_count: usize,
) -> PyResult<Option<(Py<PyBytes>, usize, usize, Py<PyBytes>, usize, usize)>> {
    let (augmented, pivot_cols) = gf2_rref_packed_impl(
        pack_augmented_bytes(data.as_bytes(), row_count, col_count, true)?,
        row_count,
        col_count,
    );
    if pivot_cols.len() != row_count {
        return Ok(None);
    }

    let mut right_inverse = vec![0u8; col_count * row_count];
    for (row, &pivot_col) in pivot_cols.iter().enumerate() {
        for col in 0..row_count {
            right_inverse[pivot_col * row_count + col] =
                get_bit(&augmented.rows[row], col_count + col) as u8;
        }
    }

    let mut is_pivot = vec![false; col_count];
    for &pivot_col in &pivot_cols {
        is_pivot[pivot_col] = true;
    }
    let free_cols: Vec<usize> = (0..col_count).filter(|&col| !is_pivot[col]).collect();

    let mut kernel = vec![0u8; col_count * free_cols.len()];
    for (basis_col, &free_col) in free_cols.iter().enumerate() {
        kernel[free_col * free_cols.len() + basis_col] = 1;
        for (row, &pivot_col) in pivot_cols.iter().enumerate() {
            kernel[pivot_col * free_cols.len() + basis_col] =
                get_bit(&augmented.rows[row], free_col) as u8;
        }
    }

    Ok(Some((
        PyBytes::new(py, &right_inverse).unbind(),
        col_count,
        row_count,
        PyBytes::new(py, &kernel).unbind(),
        col_count,
        free_cols.len(),
    )))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gf2_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(gf2_inverse_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(gf2_right_inverse_and_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(gf2_right_inverse_and_kernel_bytes, m)?)?;
    Ok(())
}
