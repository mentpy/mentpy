// Copyright 2026 Luis Mantilla
// Licensed under the Apache License, Version 2.0.
// See <http://www.apache.org/licenses/LICENSE-2.0> for details.
//! Runtime-loaded cuPauliProp 0.5 interface. No CUDA link or toolkit is needed
//! to build the CPU wheel. Resources are owned here and outlive each FFI call.

use libloading::Library;
use num_complex::Complex64 as C;
use pyo3::exceptions::{PyMemoryError, PyRuntimeError};
use pyo3::PyResult;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr::null_mut;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

type Ptr = *mut c_void;
type Status = i32;

// ABI declarations follow the public cuPauliProp 0.5 C API. The library version
// is checked before use. Truncation pointers are always null with count zero.
macro_rules! api {
    ($($field:ident: $symbol:literal ($($arg:ty),*) -> $result:ty;)*) => {
        struct Api {
            $($field: unsafe extern "C" fn($($arg),*) -> $result,)*
            cuda_malloc: unsafe extern "C" fn(*mut Ptr, usize) -> Status,
            cuda_free: unsafe extern "C" fn(Ptr) -> Status,
            cuda_copy: unsafe extern "C" fn(Ptr, *const c_void, usize, i32) -> Status,
            cuda_sync: unsafe extern "C" fn() -> Status,
            cuda_get: unsafe extern "C" fn(*mut i32) -> Status,
            cuda_set: unsafe extern "C" fn(i32) -> Status,
            cuda_error: unsafe extern "C" fn(Status) -> *const c_char,
            _cupp: Library,
            _runtime: Library,
        }
        impl Api {
            unsafe fn load(path: &str, runtime: &str) -> PyResult<Self> {
                // SAFETY: paths are resolved from installed NVIDIA packages by
                // the Python interface. Libraries stay owned alongside symbols.
                let cupp = Library::new(path).map_err(error)?;
                let runtime = Library::new(runtime).map_err(error)?;
                let api = Self {
                    $($field: *cupp.get(concat!($symbol, "\0").as_bytes()).map_err(error)?,)*
                    cuda_malloc: *runtime.get(b"cudaMalloc\0").map_err(error)?,
                    cuda_free: *runtime.get(b"cudaFree\0").map_err(error)?,
                    cuda_copy: *runtime.get(b"cudaMemcpy\0").map_err(error)?,
                    cuda_sync: *runtime.get(b"cudaDeviceSynchronize\0").map_err(error)?,
                    cuda_get: *runtime.get(b"cudaGetDevice\0").map_err(error)?,
                    cuda_set: *runtime.get(b"cudaSetDevice\0").map_err(error)?,
                    cuda_error: *runtime.get(b"cudaGetErrorString\0").map_err(error)?,
                    _cupp: cupp, _runtime: runtime,
                };
                if (api.version)() / 100 != 5 {
                    return Err(PyRuntimeError::new_err("This interface requires cuPauliProp 0.5.x."));
                }
                Ok(api)
            }
        }
    };
}

api! {
    version: "cupaulipropGetVersion" () -> usize;
    error: "cupaulipropGetErrorString" (Status) -> *const c_char;
    create: "cupaulipropCreate" (*mut Ptr) -> Status;
    destroy: "cupaulipropDestroy" (Ptr) -> Status;
    workspace: "cupaulipropCreateWorkspaceDescriptor" (Ptr, *mut Ptr) -> Status;
    free_workspace: "cupaulipropDestroyWorkspaceDescriptor" (Ptr) -> Status;
    workspace_size: "cupaulipropWorkspaceGetMemorySize" (Ptr, Ptr, i32, i32, *mut i64) -> Status;
    workspace_set: "cupaulipropWorkspaceSetMemory" (Ptr, Ptr, i32, i32, Ptr, i64) -> Status;
    expansion: "cupaulipropCreatePauliExpansion" (Ptr, i32, Ptr, i64, Ptr, i64, i32, i64, i32, i32, *mut Ptr) -> Status;
    free_expansion: "cupaulipropDestroyPauliExpansion" (Ptr) -> Status;
    num_terms: "cupaulipropPauliExpansionGetNumTerms" (Ptr, Ptr, *mut i64) -> Status;
    view: "cupaulipropPauliExpansionGetContiguousRange" (Ptr, Ptr, i64, i64, *mut Ptr) -> Status;
    free_view: "cupaulipropDestroyPauliExpansionView" (Ptr) -> Status;
    clifford: "cupaulipropCreateCliffordGateOperator" (Ptr, i32, *const i32, *mut Ptr) -> Status;
    rotation: "cupaulipropCreatePauliRotationGateOperator" (Ptr, f64, i32, *const i32, *const i32, *mut Ptr) -> Status;
    free_gate: "cupaulipropDestroyOperator" (Ptr) -> Status;
    cotangent: "cupaulipropQuantumOperatorAttachCotangentBuffer" (Ptr, Ptr, Ptr, i64, i32, i32) -> Status;
    prepare: "cupaulipropPauliExpansionViewPrepareOperatorApplication" (Ptr, Ptr, Ptr, i32, i32, i32, *const c_void, i64, *mut i64, *mut i64, Ptr) -> Status;
    apply: "cupaulipropPauliExpansionViewComputeOperatorApplication" (Ptr, Ptr, Ptr, Ptr, i32, i32, i32, i32, *const c_void, Ptr, Ptr) -> Status;
    prepare_trace: "cupaulipropPauliExpansionViewPrepareTraceWithZeroState" (Ptr, Ptr, i64, Ptr) -> Status;
    trace: "cupaulipropPauliExpansionViewComputeTraceWithZeroState" (Ptr, Ptr, Ptr, *mut f64, Ptr, Ptr) -> Status;
    prepare_trace_grad: "cupaulipropPauliExpansionViewPrepareTraceWithZeroStateBackwardDiff" (Ptr, Ptr, i64, *mut i64, *mut i64, Ptr) -> Status;
    trace_grad: "cupaulipropPauliExpansionViewComputeTraceWithZeroStateBackwardDiff" (Ptr, Ptr, *const c_void, *const f64, Ptr, Ptr, Ptr) -> Status;
    prepare_grad: "cupaulipropPauliExpansionViewPrepareOperatorApplicationBackwardDiff" (Ptr, Ptr, Ptr, Ptr, i32, i32, i32, *const c_void, i64, *mut i64, *mut i64, Ptr) -> Status;
    apply_grad: "cupaulipropPauliExpansionViewComputeOperatorApplicationBackwardDiff" (Ptr, Ptr, Ptr, Ptr, Ptr, i32, i32, i32, i32, *const c_void, Ptr, Ptr) -> Status;
}

fn error(value: impl std::fmt::Display) -> pyo3::PyErr {
    PyRuntimeError::new_err(format!("CUDA Pauli propagation: {value}"))
}

impl Api {
    fn shared(path: &str, runtime: &str) -> PyResult<Arc<Self>> {
        type Cache = Mutex<HashMap<(String, String), Arc<Api>>>;
        static CACHE: OnceLock<Cache> = OnceLock::new();
        let mut cache = CACHE.get_or_init(Default::default).lock().map_err(error)?;
        let key = (path.to_owned(), runtime.to_owned());
        if let Some(api) = cache.get(&key) {
            return Ok(api.clone());
        }
        // SAFETY: ABI version and all symbols are validated in load().
        let api = Arc::new(unsafe { Self::load(path, runtime)? });
        cache.insert(key, api.clone());
        Ok(api)
    }

    fn check(&self, status: Status) -> PyResult<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: the library returns a static NUL-terminated error string.
        Err(error(
            unsafe { CStr::from_ptr((self.error)(status)) }.to_string_lossy(),
        ))
    }

    fn cuda(&self, status: Status) -> PyResult<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: the runtime returns a static NUL-terminated error string.
        Err(error(
            unsafe { CStr::from_ptr((self.cuda_error)(status)) }.to_string_lossy(),
        ))
    }
}

struct Budget {
    used: Cell<usize>,
    limit: usize,
}

struct Buffer {
    api: Arc<Api>,
    ptr: Ptr,
    size: usize,
    budget: Rc<Budget>,
}

impl Buffer {
    fn new(api: &Arc<Api>, budget: &Rc<Budget>, size: usize) -> PyResult<Self> {
        let size = size.max(16);
        if size > budget.limit.saturating_sub(budget.used.get()) {
            return Err(PyMemoryError::new_err(
                "CUDA Pauli propagation exceeds max_memory.",
            ));
        }
        let mut ptr = null_mut();
        // SAFETY: ptr is an output slot. CUDA allocates at least size bytes.
        api.cuda(unsafe { (api.cuda_malloc)(&mut ptr, size) })?;
        budget.used.set(budget.used.get() + size);
        Ok(Self {
            api: api.clone(),
            ptr,
            size,
            budget: budget.clone(),
        })
    }

    fn write<T: bytemuck::Pod>(&self, values: &[T]) -> PyResult<()> {
        let size = std::mem::size_of_val(values);
        if size > self.size {
            return Err(error("GPU buffer is too small."));
        }
        // SAFETY: source/destination ranges have size bytes. Copy is blocking.
        self.api
            .cuda(unsafe { (self.api.cuda_copy)(self.ptr, values.as_ptr().cast(), size, 1) })
    }

    fn read<T: bytemuck::Pod>(&self, values: &mut [T]) -> PyResult<()> {
        let size = std::mem::size_of_val(values);
        if size > self.size {
            return Err(error("GPU buffer is too small."));
        }
        // SAFETY: both ranges have size bytes, Pod permits all bit patterns in
        // destination storage, and the device-to-host copy is synchronous.
        self.api
            .cuda(unsafe { (self.api.cuda_copy)(values.as_mut_ptr().cast(), self.ptr, size, 2) })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: each allocation is freed once, after its last library use.
        unsafe {
            (self.api.cuda_free)(self.ptr);
        }
        self.budget.used.set(self.budget.used.get() - self.size);
    }
}

struct Context {
    api: Arc<Api>,
    handle: Ptr,
    workspace: Ptr,
    previous: i32,
    budget: Rc<Budget>,
    scratch: RefCell<Option<Buffer>>,
}

impl Context {
    fn new(path: &str, runtime: &str, device: i32, limit: usize) -> PyResult<Self> {
        let api = Api::shared(path, runtime)?;
        let mut previous = 0;
        // SAFETY: output slot is valid; CUDA validates the device index.
        unsafe {
            api.cuda((api.cuda_get)(&mut previous))?;
            api.cuda((api.cuda_set)(device))?;
        }
        let mut ctx = Self {
            api,
            handle: null_mut(),
            workspace: null_mut(),
            previous,
            budget: Rc::new(Budget {
                used: Cell::new(0),
                limit,
            }),
            scratch: RefCell::new(None),
        };
        // SAFETY: handles are owned output slots; Drop handles partial creation.
        unsafe {
            ctx.api.check((ctx.api.create)(&mut ctx.handle))?;
            ctx.api
                .check((ctx.api.workspace)(ctx.handle, &mut ctx.workspace))?;
        }
        Ok(ctx)
    }

    fn scratch(&self) -> PyResult<()> {
        let mut size = 0;
        // SAFETY: context/workspace are live; size is an output slot.
        self.api.check(unsafe {
            (self.api.workspace_size)(self.handle, self.workspace, 0, 0, &mut size)
        })?;
        let size = usize::try_from(size).map_err(error)?;
        let mut scratch = self.scratch.borrow_mut();
        if scratch.as_ref().is_none_or(|s| s.size < size) {
            // All preceding compute calls have completed before workspace reuse.
            scratch.take();
            *scratch = Some(Buffer::new(&self.api, &self.budget, size)?);
        }
        let buffer = scratch.as_ref().unwrap();
        // SAFETY: buffer outlives compute and is sized by the prepare API.
        self.api.check(unsafe {
            (self.api.workspace_set)(
                self.handle,
                self.workspace,
                0,
                0,
                buffer.ptr,
                buffer.size as i64,
            )
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: borrowed views/gates/expansions have been dropped. Restore the
        // caller's current device only after freeing this context's resources.
        unsafe {
            (self.api.cuda_sync)();
            if !self.workspace.is_null() {
                (self.api.free_workspace)(self.workspace);
            }
            self.scratch.get_mut().take();
            if !self.handle.is_null() {
                (self.api.destroy)(self.handle);
            }
            (self.api.cuda_set)(self.previous);
        }
    }
}

struct Expansion<'a> {
    ctx: &'a Context,
    ptr: Ptr,
    _words: Buffer,
    _coeff: Buffer,
}

impl<'a> Expansion<'a> {
    fn new(
        ctx: &'a Context,
        n: usize,
        words: i64,
        coeff: i64,
        initial: Option<(&[u64], &[f64])>,
    ) -> PyResult<Self> {
        let words = Buffer::new(
            &ctx.api,
            &ctx.budget,
            usize::try_from(words).map_err(error)?,
        )?;
        let coeff = Buffer::new(
            &ctx.api,
            &ctx.budget,
            usize::try_from(coeff).map_err(error)?,
        )?;
        let mut count = 0;
        if let Some((w, c)) = initial {
            words.write(w)?;
            coeff.write(c)?;
            count = c.len();
        }
        let mut ptr = null_mut();
        // SAFETY: owned, aligned CUDA buffers outlive this handle. Coefficients
        // are f64 (CUDA_R_64F=1), with count initialized terms and no duplicates.
        ctx.api.check(unsafe {
            (ctx.api.expansion)(
                ctx.handle,
                n as i32,
                words.ptr,
                words.size as i64,
                coeff.ptr,
                coeff.size as i64,
                1,
                count as i64,
                0,
                0,
                &mut ptr,
            )
        })?;
        Ok(Self {
            ctx,
            ptr,
            _words: words,
            _coeff: coeff,
        })
    }

    fn len(&self) -> PyResult<usize> {
        let mut size = 0;
        // SAFETY: live expansion and valid output slot.
        self.ctx
            .api
            .check(unsafe { (self.ctx.api.num_terms)(self.ctx.handle, self.ptr, &mut size) })?;
        usize::try_from(size).map_err(error)
    }

    fn view(&self) -> PyResult<View<'_>> {
        let mut ptr = null_mut();
        // SAFETY: the range covers the live expansion; the view borrows it.
        self.ctx.api.check(unsafe {
            (self.ctx.api.view)(self.ctx.handle, self.ptr, 0, self.len()? as i64, &mut ptr)
        })?;
        Ok(View {
            expansion: self,
            ptr,
        })
    }
}

impl Drop for Expansion<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.ctx.api.free_expansion)(self.ptr);
        }
    }
}

struct View<'a> {
    expansion: &'a Expansion<'a>,
    ptr: Ptr,
}
impl Drop for View<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.expansion.ctx.api.free_view)(self.ptr);
        }
    }
}

struct Gate<'a> {
    ctx: &'a Context,
    ptr: Ptr,
    parameter: Option<usize>,
    cotangent: Box<UnsafeCell<f64>>,
}

impl<'a> Gate<'a> {
    fn new(
        ctx: &'a Context,
        kind: i32,
        wires: &[i32],
        rotation: Option<(Option<usize>, f64)>,
    ) -> PyResult<Self> {
        let mut gate = Self {
            ctx,
            ptr: null_mut(),
            parameter: rotation.and_then(|(i, _)| i),
            cotangent: Box::new(UnsafeCell::new(0.)),
        };
        // SAFETY: wires have the gate's required arity and are in range. The
        // library copies gate parameters; cotangent stays boxed until destruction.
        unsafe {
            if let Some((_, angle)) = rotation {
                ctx.api.check((ctx.api.rotation)(
                    ctx.handle,
                    angle,
                    1,
                    wires.as_ptr(),
                    [3].as_ptr(),
                    &mut gate.ptr,
                ))?;
                ctx.api.check((ctx.api.cotangent)(
                    ctx.handle,
                    gate.ptr,
                    gate.cotangent.get().cast(),
                    8,
                    1,
                    1,
                ))?;
            } else {
                ctx.api.check((ctx.api.clifford)(
                    ctx.handle,
                    kind,
                    wires.as_ptr(),
                    &mut gate.ptr,
                ))?;
            }
        }
        Ok(gate)
    }
}

impl Drop for Gate<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.ctx.api.free_gate)(self.ptr);
            }
        }
    }
}

pub struct Request<'a> {
    pub n: usize,
    pub outputs: &'a [usize],
    pub operations: &'a [(u8, usize, usize)],
    pub angles: &'a [f64],
    pub terms: &'a [(String, f64)],
    pub constant: f64,
    pub gradient: bool,
    pub max_terms: usize,
    pub active: &'a [bool],
    pub input: Option<&'a [C]>,
    pub n_inputs: usize,
}

fn input_mean(word: &[u64], n: usize, n_inputs: usize, state: &[C]) -> f64 {
    let packed = n.div_ceil(64);
    // Only ancillary roots were transformed by H, so they are traced in |0>.
    if (n_inputs..n).any(|q| word[q / 64] >> (q % 64) & 1 != 0) {
        return 0.;
    }
    let (mut x, mut z, mut ys) = (0usize, 0usize, 0usize);
    for q in 0..n_inputs {
        let xb = (word[q / 64] >> (q % 64) & 1) as usize;
        let zb = (word[packed + q / 64] >> (q % 64) & 1) as usize;
        x = (x << 1) | xb;
        z = (z << 1) | zb;
        ys += xb & zb;
    }
    let phase = [
        C::new(1., 0.),
        C::new(0., 1.),
        C::new(-1., 0.),
        C::new(0., -1.),
    ][ys % 4];
    state
        .iter()
        .enumerate()
        .map(|(i, amp)| {
            let sign = if (i & z).count_ones() % 2 == 0 {
                1.
            } else {
                -1.
            };
            (state[i ^ x].conj() * phase * sign * amp).re
        })
        .sum()
}

pub fn evaluate(
    request: Request<'_>,
    path: &str,
    runtime: &str,
    device: i32,
    memory: usize,
) -> PyResult<(f64, Vec<f64>, usize)> {
    let Request {
        n,
        outputs,
        operations,
        angles,
        terms,
        constant,
        gradient,
        max_terms,
        active,
        input,
        n_inputs,
    } = request;
    let ctx = Context::new(path, runtime, device, memory)?;
    if n == 0 {
        return Ok((
            constant + terms.iter().map(|(_, c)| c).sum::<f64>(),
            if gradient {
                vec![0.; angles.len()]
            } else {
                vec![]
            },
            usize::from(constant != 0. || !terms.is_empty()),
        ));
    }
    let a = &ctx.api;
    let packed = n.div_ceil(64);
    let mut words = Vec::new();
    let mut coeff = Vec::new();
    let mut merged = std::collections::BTreeMap::new();
    for (word, coefficient) in terms
        .iter()
        .map(|(s, c)| (s.as_str(), *c))
        .chain(std::iter::once(("", constant)))
    {
        if coefficient == 0. {
            continue;
        }
        let mut bits = vec![0u64; 2 * packed];
        for (letter, &wire) in word.bytes().zip(outputs) {
            if letter == b'X' || letter == b'Y' {
                bits[wire / 64] |= 1 << (wire % 64);
            }
            if letter == b'Z' || letter == b'Y' {
                bits[packed + wire / 64] |= 1 << (wire % 64);
            }
        }
        *merged.entry(bits).or_insert(0.) += coefficient;
    }
    for (bits, coefficient) in merged {
        if coefficient != 0. {
            words.extend(bits);
            coeff.push(coefficient);
        }
    }
    if coeff.is_empty() {
        return Ok((
            0.,
            if gradient {
                vec![0.; angles.len()]
            } else {
                vec![]
            },
            0,
        ));
    }
    if coeff.len() > max_terms {
        return Err(PyMemoryError::new_err("Pauli expansion exceeds max_terms."));
    }
    let mut gates = Vec::new();
    for wire in if input.is_some() { n_inputs } else { 0 }..n {
        gates.push(Gate::new(&ctx, 4, &[wire as i32], None)?);
    }
    for &(kind, wire, operand) in operations {
        if kind == 1 {
            gates.push(Gate::new(&ctx, 9, &[wire as i32, operand as i32], None)?);
        } else {
            let quadrant = if active[operand] {
                None
            } else {
                super::tensor::clifford(-angles[operand])
            };
            if let Some(quadrant) = quadrant {
                // Non-trainable Clifford rotations have exact signed-word
                // actions. Do not grow tiny sine/cosine residual branches.
                if quadrant >= 2 {
                    gates.push(Gate::new(&ctx, 3, &[wire as i32], None)?);
                }
                if quadrant % 2 == 1 {
                    gates.push(Gate::new(&ctx, 5, &[wire as i32], None)?);
                }
            } else {
                gates.push(Gate::new(
                    &ctx,
                    0,
                    &[wire as i32],
                    Some((active[operand].then_some(operand), -angles[operand])),
                )?);
            }
            gates.push(Gate::new(&ctx, 4, &[wire as i32], None)?);
        }
    }
    gates.reverse();
    let initial = Expansion::new(
        &ctx,
        n,
        (words.len() * 8) as i64,
        (coeff.len() * 8) as i64,
        Some((&words, &coeff)),
    )?;
    let mut states = vec![initial];
    let mut peak = coeff.len();
    for gate in &gates {
        let view = states.last().unwrap().view()?;
        let (mut w, mut c) = (0, 0);
        // SAFETY: every handle and buffer below is live, prepared capacities are
        // checked before compute, and no truncation strategies are supplied.
        unsafe {
            a.check((a.prepare)(
                ctx.handle,
                view.ptr,
                gate.ptr,
                0,
                0,
                0,
                null_mut(),
                memory as i64,
                &mut w,
                &mut c,
                ctx.workspace,
            ))?;
        }
        let output = Expansion::new(&ctx, n, w, c, None)?;
        ctx.scratch()?;
        unsafe {
            a.check((a.apply)(
                ctx.handle,
                view.ptr,
                output.ptr,
                gate.ptr,
                1,
                0,
                0,
                0,
                null_mut(),
                ctx.workspace,
                null_mut(),
            ))?;
        }
        let size = output.len()?;
        peak = peak.max(size);
        if size > max_terms {
            return Err(PyMemoryError::new_err(
                "Pauli expansion exceeds max_terms; no terms were truncated.",
            ));
        }
        drop(view);
        if !gradient {
            states.clear();
        }
        states.push(output);
    }
    let final_state = states.last().unwrap();
    let (value, mut cotangent) = if let Some(input) = input {
        // General entangled input readout is native CPU work in its supplied
        // 2**n_inputs representation. Propagation and backprop remain on GPU.
        // This avoids expanding that input into a 4**n_inputs density operator.
        let count = final_state.len()?;
        let mut words = vec![0u64; count * 2 * packed];
        let mut coeff = vec![0f64; count];
        final_state._words.read(&mut words)?;
        final_state._coeff.read(&mut coeff)?;
        let means: Vec<_> = words
            .chunks_exact(2 * packed)
            .map(|word| input_mean(word, n, n_inputs, input))
            .collect();
        let value: f64 = coeff.iter().zip(&means).map(|(c, mean)| c * mean).sum();
        if !value.is_finite() {
            return Err(error("Non-finite GPU expectation."));
        }
        if !gradient {
            return Ok((value, vec![], peak));
        }
        (
            value,
            Expansion::new(
                &ctx,
                n,
                (words.len() * 8) as i64,
                (means.len() * 8) as i64,
                Some((&words, &means)),
            )?,
        )
    } else {
        let final_view = final_state.view()?;
        unsafe {
            a.check((a.prepare_trace)(
                ctx.handle,
                final_view.ptr,
                memory as i64,
                ctx.workspace,
            ))?;
        }
        ctx.scratch()?;
        let (mut significand, mut exponent) = (0f64, 0f64);
        unsafe {
            a.check((a.trace)(
                ctx.handle,
                final_view.ptr,
                (&mut significand as *mut f64).cast(),
                &mut exponent,
                ctx.workspace,
                null_mut(),
            ))?;
            // Trace is asynchronous; host outputs must not be read before sync.
            a.cuda((a.cuda_sync)())?;
        }
        let value = significand * exponent.exp2();
        if !value.is_finite() {
            return Err(error("Non-finite GPU expectation."));
        }
        if !gradient {
            return Ok((value, vec![], peak));
        }
        let (mut w, mut c) = (0, 0);
        unsafe {
            a.check((a.prepare_trace_grad)(
                ctx.handle,
                final_view.ptr,
                memory as i64,
                &mut w,
                &mut c,
                ctx.workspace,
            ))?;
        }
        let cotangent = Expansion::new(&ctx, n, w, c, None)?;
        ctx.scratch()?;
        let cot_s = exponent.exp2();
        let cot_p = value * std::f64::consts::LN_2;
        unsafe {
            a.check((a.trace_grad)(
                ctx.handle,
                final_view.ptr,
                (&cot_s as *const f64).cast(),
                &cot_p,
                cotangent.ptr,
                ctx.workspace,
                null_mut(),
            ))?;
            a.cuda((a.cuda_sync)())?;
        }
        drop(final_view);
        (value, cotangent)
    };
    states.pop();
    let (mut w, mut c) = (0, 0);
    for (i, gate) in gates.iter().enumerate().rev() {
        let input = states[i].view()?;
        let cot_out = cotangent.view()?;
        unsafe {
            a.check((a.prepare_grad)(
                ctx.handle,
                input.ptr,
                cot_out.ptr,
                gate.ptr,
                0,
                0,
                0,
                null_mut(),
                memory as i64,
                &mut w,
                &mut c,
                ctx.workspace,
            ))?;
        }
        let cot_in = Expansion::new(&ctx, n, w, c, None)?;
        ctx.scratch()?;
        unsafe {
            a.check((a.apply_grad)(
                ctx.handle,
                input.ptr,
                cot_out.ptr,
                cot_in.ptr,
                gate.ptr,
                1,
                0,
                0,
                0,
                null_mut(),
                ctx.workspace,
                null_mut(),
            ))?;
        }
        drop(cot_out);
        drop(input);
        states.pop();
        cotangent = cot_in;
    }
    let mut derivative = vec![0.; angles.len()];
    for gate in gates {
        if let Some(index) = gate.parameter {
            // SAFETY: backward compute is blocking. UnsafeCell permits the
            // library to update the boxed buffer while a Gate is shared.
            derivative[index] -= unsafe { *gate.cotangent.get() };
        }
    }
    if !value.is_finite() || derivative.iter().any(|v| !v.is_finite()) {
        return Err(error("Non-finite GPU expectation or gradient."));
    }
    Ok((value, derivative, peak))
}
