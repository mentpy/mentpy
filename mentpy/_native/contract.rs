// Copyright 2026 Luis Mantilla
// Licensed under the Apache License, Version 2.0.
// See <http://www.apache.org/licenses/LICENSE-2.0> for details.
//! Binary tensor-network planning and differentiable complex contractions.

use candle_core::{Result, Tensor};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

#[derive(Clone)]
pub struct Complex {
    pub re: Tensor,
    pub im: Tensor,
}

impl Complex {
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        Ok(Self {
            re: self.re.reshape(shape)?,
            im: self.im.reshape(shape)?,
        })
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        Ok(Self {
            re: self.re.permute(axes)?,
            im: self.im.permute(axes)?,
        })
    }

    pub fn rescale(&self) -> Result<Self> {
        let scale = (self.re.sqr()? + self.im.sqr()?)?
            .max_all()?
            .sqrt()?
            .detach();
        let valid = scale.gt(f64::EPSILON)?;
        let divisor = scale.clamp(f64::EPSILON, f64::MAX)?;
        let re = self.re.broadcast_div(&divisor)?;
        let im = self.im.broadcast_div(&divisor)?;
        let valid = valid.broadcast_as(re.shape())?;
        Ok(Self {
            re: valid.where_cond(&re, &re.zeros_like()?)?,
            im: valid.where_cond(&im, &im.zeros_like()?)?,
        })
    }
}

pub struct Step {
    pub left: usize,
    pub right: usize,
    pub left_axes: Vec<usize>,
    pub right_axes: Vec<usize>,
    pub left_shape: Vec<usize>,
    pub right_shape: Vec<usize>,
    shape: Vec<usize>,
}

impl Step {
    pub fn apply(&self, a: &Complex, b: &Complex) -> Result<Complex> {
        let a = a.permute(&self.left_axes)?.reshape(&self.left_shape)?;
        let b = b.permute(&self.right_axes)?.reshape(&self.right_shape)?;
        let re = (a.re.matmul(&b.re)? - a.im.matmul(&b.im)?)?;
        let im = (a.re.matmul(&b.im)? + a.im.matmul(&b.re)?)?;
        Complex { re, im }.reshape(&self.shape)?.rescale()
    }
}

pub struct Plan {
    pub steps: Vec<Step>,
    pub output_axes: Vec<usize>,
    pub peak_elements: usize,
}

fn pair_labels(
    left: &[usize],
    right: &[usize],
    counts: &BTreeMap<usize, usize>,
    outputs: &BTreeSet<usize>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let shared: BTreeSet<_> = left.iter().copied().filter(|x| right.contains(x)).collect();
    let (batch, sum): (Vec<_>, Vec<_>) = shared
        .into_iter()
        .partition(|x| counts[x] > 2 || outputs.contains(x));
    let a: Vec<_> = left
        .iter()
        .copied()
        .filter(|x| !right.contains(x))
        .collect();
    let b: Vec<_> = right
        .iter()
        .copied()
        .filter(|x| !left.contains(x))
        .collect();
    (batch, a, sum, b)
}

fn pair_cost(
    left: &[usize],
    right: &[usize],
    counts: &BTreeMap<usize, usize>,
    outputs: &BTreeSet<usize>,
) -> (usize, usize) {
    let (batch, a, sum, b) = pair_labels(left, right, counts, outputs);
    let rank = batch.len() + a.len() + b.len();
    (rank, rank + sum.len())
}

fn axes(labels: &[usize], order: &[usize]) -> Vec<usize> {
    order
        .iter()
        .map(|x| labels.iter().position(|y| x == y).unwrap())
        .collect()
}

fn size(rank: usize, limit: usize) -> Result<usize> {
    let value = 1usize.checked_shl(rank as u32).filter(|&n| n <= limit);
    value.ok_or_else(|| candle_core::Error::Msg(format!("Tensor contraction exceeds max_elements={limit}; choose another backend or increase the limit.")))
}

impl Plan {
    pub fn new(indices: &[Vec<usize>], output: &[usize], limit: usize) -> Result<Self> {
        let output_set: BTreeSet<_> = output.iter().copied().collect();
        let mut counts = BTreeMap::new();
        let mut owners: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        let mut live: BTreeMap<usize, Vec<usize>> = indices.iter().cloned().enumerate().collect();
        let mut peak_elements = 1;
        for (i, labels) in indices.iter().enumerate() {
            peak_elements = peak_elements.max(size(labels.len(), limit)?);
            for &label in labels {
                *counts.entry(label).or_insert(0) += 1;
                owners.entry(label).or_default().insert(i);
            }
        }
        let mut heap = BinaryHeap::new();
        let mut pairs = BTreeSet::new();
        for group in owners.values() {
            for &i in group {
                for &j in group.range(i + 1..) {
                    pairs.insert((i, j));
                }
            }
        }
        for (i, j) in pairs {
            let (rank, cost) = pair_cost(&live[&i], &live[&j], &counts, &output_set);
            heap.push(Reverse((rank, cost, i, j)));
        }
        let mut steps = Vec::new();
        while live.len() > 1 {
            let (i, j) = loop {
                if let Some(Reverse((rank, cost, i, j))) = heap.pop() {
                    let (Some(a), Some(b)) = (live.get(&i), live.get(&j)) else {
                        continue;
                    };
                    let new_cost = pair_cost(a, b, &counts, &output_set);
                    if (rank, cost) != new_cost {
                        heap.push(Reverse((new_cost.0, new_cost.1, i, j)));
                        continue;
                    }
                    break (i, j);
                }
                // Disconnected components need an outer product only at the end.
                let mut smallest: Vec<_> = live.iter().map(|(&i, x)| (x.len(), i)).collect();
                smallest.sort_unstable();
                break (smallest[0].1, smallest[1].1);
            };
            let a = live.remove(&i).unwrap();
            let b = live.remove(&j).unwrap();
            let (batch, left, sum, right) = pair_labels(&a, &b, &counts, &output_set);
            let labels: Vec<_> = batch.iter().chain(&left).chain(&right).copied().collect();
            peak_elements = peak_elements.max(size(labels.len(), limit)?);
            let left_order: Vec<_> = batch.iter().chain(&left).chain(&sum).copied().collect();
            let right_order: Vec<_> = batch.iter().chain(&sum).chain(&right).copied().collect();
            let batch_size = size(batch.len(), limit)?;
            steps.push(Step {
                left: i,
                right: j,
                left_axes: axes(&a, &left_order),
                right_axes: axes(&b, &right_order),
                left_shape: vec![
                    batch_size,
                    size(left.len(), limit)?,
                    size(sum.len(), limit)?,
                ],
                right_shape: vec![
                    batch_size,
                    size(sum.len(), limit)?,
                    size(right.len(), limit)?,
                ],
                shape: vec![2; labels.len()],
            });
            let next = indices.len() + steps.len() - 1;
            for label in a.iter().chain(&b) {
                *counts.get_mut(label).unwrap() -= 1;
                owners.get_mut(label).unwrap().remove(&i);
                owners.get_mut(label).unwrap().remove(&j);
            }
            let mut neighbors = BTreeSet::new();
            for &label in &labels {
                *counts.get_mut(&label).unwrap() += 1;
                neighbors.extend(owners[&label].iter().copied());
                owners.get_mut(&label).unwrap().insert(next);
            }
            for neighbor in neighbors {
                let (rank, cost) = pair_cost(&live[&neighbor], &labels, &counts, &output_set);
                heap.push(Reverse((rank, cost, neighbor, next)));
            }
            live.insert(next, labels);
        }
        let final_labels = live.values().next().cloned().unwrap_or_default();
        if final_labels.len() != output.len() || !output.iter().all(|x| final_labels.contains(x)) {
            candle_core::bail!("Tensor network has unmatched internal indices.");
        }
        Ok(Self {
            steps,
            output_axes: axes(&final_labels, output),
            peak_elements,
        })
    }

    pub fn contract(&self, operands: Vec<Complex>) -> Result<Complex> {
        let mut values: Vec<_> = operands.into_iter().map(Some).collect();
        for step in &self.steps {
            let a = values[step.left].take().unwrap();
            let b = values[step.right].take().unwrap();
            values.push(Some(step.apply(&a, &b)?));
        }
        values.pop().unwrap().unwrap().permute(&self.output_axes)
    }
}
