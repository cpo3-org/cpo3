//! Right-preconditioned LSMR with callback-controlled termination.
//!
//! Starts from zero and calls the callback before iteration and after each
//! completed step. The caller supplies all acceptance criteria; there are no
//! built-in tolerance checks. Iteration-limit and breakdown exits do not imply
//! convergence.
//!
//! Adapted from faer 0.24.4, with callback termination inspired by PR #341:
//! https://codeberg.org/sarah-quinones/faer/pulls/341

/*
MIT License

Copyright (c) 2026 sarah quiñones el kazdadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#![allow(non_snake_case)] // Preserve upstream mathematical notation.

use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::{householder, matmul::matmul, qr, temp_mat_scratch, temp_mat_uninit};
use faer::mat::AsMatMut;
use faer::matrix_free::{BiLinOp, BiPrecond};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::traits::{
    ComplexField,
    math_utils::{one, zero},
};
use faer::{Accum, Conj, MatMut, MatRef, Par};
use std::ops::ControlFlow;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum LsmrStop<B> {
    Callback(B),
    IterationLimit,
    Breakdown,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LsmrInfo<B> {
    pub iter_count: usize,
    pub stop: LsmrStop<B>,
}

/// computes the layout of required workspace for executing the lsmr
/// algorithm
pub(crate) fn lsmr_scratch<T: ComplexField>(
    right_precond: impl BiPrecond<T>,
    mat: impl BiLinOp<T>,
    rhs_ncols: usize,
    par: Par,
) -> StackReq {
    fn implementation<T: ComplexField>(
        M: &dyn BiPrecond<T>,
        A: &dyn BiLinOp<T>,
        rhs_ncols: usize,
        par: Par,
    ) -> StackReq {
        let m = A.nrows();
        let n = A.ncols();
        let mut k = rhs_ncols;
        if m == 0 || n == 0 || k == 0 {
            return StackReq::EMPTY;
        }
        assert!(k < isize::MAX as usize);
        if k > n {
            k = k.checked_next_multiple_of(n).unwrap();
        }
        assert!(k < isize::MAX as usize);
        let s = Ord::min(k, Ord::min(n, m));
        let mk = temp_mat_scratch::<T>(m, k);
        let nk = temp_mat_scratch::<T>(n, k);
        let ss = temp_mat_scratch::<T>(s, s);
        let ss2 = temp_mat_scratch::<T>(2 * s, 2 * s);
        let sk = temp_mat_scratch::<T>(s, k);
        let sk2 = temp_mat_scratch::<T>(2 * s, 2 * k);
        let ms_bs = qr::no_pivoting::factor::recommended_block_size::<T>(m, s);
        let ns_bs = qr::no_pivoting::factor::recommended_block_size::<T>(n, s);
        let ss_bs = qr::no_pivoting::factor::recommended_block_size::<T>(2 * s, 2 * s);
        let AT = A.transpose_apply_scratch(k, par);
        let A = A.apply_scratch(k, par);
        let MT = M.transpose_apply_in_place_scratch(k, par);
        let M = M.apply_in_place_scratch(k, par);
        let m_qr = StackReq::any_of(&[
            temp_mat_scratch::<T>(ms_bs, s),
            qr::no_pivoting::factor::qr_in_place_scratch::<T>(m, s, ms_bs, par, Default::default()),
            householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                m, ms_bs, s,
            ),
        ]);
        let n_qr = StackReq::any_of(&[
            temp_mat_scratch::<T>(ns_bs, s),
            qr::no_pivoting::factor::qr_in_place_scratch::<T>(n, s, ns_bs, par, Default::default()),
            householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                n, ns_bs, s,
            ),
        ]);
        let s_qr = StackReq::any_of(&[
            temp_mat_scratch::<T>(ss_bs, s),
            qr::no_pivoting::factor::qr_in_place_scratch::<T>(
                2 * s,
                2 * s,
                ss_bs,
                par,
                Default::default(),
            ),
            householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                2 * s,
                ss_bs,
                2 * s,
            ),
        ]);
        StackReq::all_of(&[
            mk,
            nk,
            sk,
            sk,
            sk,
            sk,
            sk,
            sk2,
            nk,
            StackReq::any_of(&[StackReq::all_of(&[mk, StackReq::any_of(&[A, M, m_qr])])]),
            StackReq::any_of(&[StackReq::all_of(&[nk, StackReq::any_of(&[AT, MT, n_qr])])]),
            ss2,
            ss,
            ss,
            ss,
            ss,
            ss,
            ss,
            StackReq::all_of(&[temp_mat_scratch::<T>(2 * s, 2 * s), s_qr]),
        ])
    }
    implementation(&right_precond, &mat, rhs_ncols, par)
}
/// Solve with right-preconditioned LSMR using only callback convergence checks.
///
/// The output is initialized to zero. The callback sees this initial guess and
/// each completed iterate. `Break(value)`
/// preserves that iterate and returns `Callback(value)`. Otherwise iteration continues
/// until the budget is exhausted or there is no search direction (`Breakdown`).
/// Neither of these exits implies convergence. Multiple right-hand sides are supported.
#[track_caller]
#[allow(clippy::too_many_arguments)] // Keep faer's calling convention.
pub(crate) fn lsmr_with_control<T: ComplexField, B>(
    out: MatMut<'_, T>,
    right_precond: impl BiPrecond<T>,
    mat: impl BiLinOp<T>,
    rhs: MatRef<'_, T>,
    max_iters: usize,
    callback: impl FnMut(MatRef<'_, T>) -> ControlFlow<B>,
    par: Par,
    stack: &mut MemStack,
) -> LsmrInfo<B> {
    #[track_caller]
    fn implementation<T: ComplexField, B>(
        mut x: MatMut<'_, T>,
        M: &impl BiPrecond<T>,
        A: &impl BiLinOp<T>,
        b: MatRef<'_, T>,
        max_iters: usize,
        callback: &mut dyn FnMut(MatRef<'_, T>) -> ControlFlow<B>,
        par: Par,
        stack: &mut MemStack,
    ) -> LsmrInfo<B> {
        fn thin_qr<T: ComplexField>(
            mut Q: MatMut<'_, T>,
            mut R: MatMut<'_, T>,
            mut mat: MatMut<'_, T>,
            par: Par,
            stack: &mut MemStack,
        ) {
            let k = R.nrows();
            let bs = qr::no_pivoting::factor::recommended_block_size::<T>(mat.nrows(), mat.ncols());
            let (mut house, mut stack) = unsafe {
                temp_mat_uninit::<T, _, _>(bs, Ord::min(mat.nrows(), mat.ncols()), stack)
            };
            let mut house = house.as_mat_mut();
            qr::no_pivoting::factor::qr_in_place(
                mat.rb_mut(),
                house.rb_mut(),
                par,
                stack.rb_mut(),
                Default::default(),
            );
            R.fill(zero());
            R.copy_from_triangular_upper(mat.rb().get(..k, ..k));
            Q.fill(zero());
            Q.rb_mut()
                .diagonal_mut()
                .column_vector_mut()
                .fill(one::<T>());
            householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                mat.rb(),
                house.rb(),
                Conj::No,
                Q.rb_mut(),
                par,
                stack.rb_mut(),
            );
        }
        let m = A.nrows();
        let n = A.ncols();
        let mut k = b.ncols();
        {
            let out = x.rb();
            let mat = A;
            let right_precond = M;
            let rhs = b;
            assert!(
                right_precond.nrows() == mat.ncols()
                    && right_precond.ncols() == mat.ncols()
                    && rhs.nrows() == mat.nrows()
                    && out.nrows() == mat.ncols()
                    && out.ncols() == rhs.ncols()
            );
        }
        x.fill(zero());
        if let ControlFlow::Break(value) = callback(x.rb()) {
            return LsmrInfo {
                iter_count: 0,
                stop: LsmrStop::Callback(value),
            };
        }
        if max_iters == 0 {
            return LsmrInfo {
                iter_count: 0,
                stop: LsmrStop::IterationLimit,
            };
        }
        if m == 0 || n == 0 || k == 0 || core::mem::size_of::<T::Unit>() == 0 {
            return LsmrInfo {
                iter_count: 0,
                stop: LsmrStop::Breakdown,
            };
        }
        debug_assert!(
            m < isize::MAX as usize && n < isize::MAX as usize && k < isize::MAX as usize
        );
        let actual_k = k;
        if k > n {
            k = k.checked_next_multiple_of(n).unwrap();
        }
        debug_assert!(k < isize::MAX as usize);
        let s = Ord::min(k, Ord::min(n, m));
        let mut stack = stack;
        let (mut u, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(m, k, stack.rb_mut()) };
        let mut u = u.as_mat_mut();
        let (mut beta, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
        let mut beta = beta.as_mat_mut();
        let (mut v, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
        let mut v = v.as_mat_mut();
        let (mut alpha, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
        let mut alpha = alpha.as_mat_mut();
        let (mut zetabar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
        let mut zetabar = zetabar.as_mat_mut();
        let (mut alphabar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
        let mut alphabar = alphabar.as_mat_mut();
        let (mut theta, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(s, k, stack.rb_mut()) };
        let mut theta = theta.as_mat_mut();
        let (mut pbar_adjoint, mut stack) =
            unsafe { temp_mat_uninit::<T, _, _>(2 * s, 2 * k, stack.rb_mut()) };
        let mut pbar_adjoint = pbar_adjoint.as_mat_mut();
        let (mut w, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
        let mut w = w.as_mat_mut();
        let (mut wbar, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
        let mut wbar = wbar.as_mat_mut();
        {
            let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(m, k, stack.rb_mut()) };
            let mut qr = qr.as_mat_mut();
            qr.rb_mut().get_mut(.., ..actual_k).copy_from(b);
            qr.rb_mut().get_mut(.., actual_k..).fill(zero());
            let mut start = 0;
            while start < k {
                let end = Ord::min(k - start, s) + start;
                let len = end - start;
                thin_qr(
                    u.rb_mut().get_mut(.., start..end),
                    beta.rb_mut().get_mut(..len, start..end),
                    qr.rb_mut().get_mut(.., start..end),
                    par,
                    stack.rb_mut(),
                );
                start = end;
            }
        }
        {
            let (mut qr, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
            let mut qr = qr.as_mat_mut();
            A.adjoint_apply(qr.rb_mut(), u.rb(), par, stack.rb_mut());
            M.adjoint_apply_in_place(qr.rb_mut(), par, stack.rb_mut());
            let mut start = 0;
            while start < k {
                let end = Ord::min(k - start, s) + start;
                let len = end - start;
                thin_qr(
                    v.rb_mut().get_mut(.., start..end),
                    alpha.rb_mut().get_mut(..len, start..end),
                    qr.rb_mut().get_mut(.., start..end),
                    par,
                    stack.rb_mut(),
                );
                start = end;
            }
        }
        zetabar.fill(zero());
        let mut start = 0;
        while start < k {
            let end = Ord::min(k - start, s) + start;
            let len = end - start;
            matmul(
                zetabar.rb_mut().get_mut(..len, start..end),
                Accum::Replace,
                alpha.rb().get(..len, start..end),
                beta.rb().get(..len, start..end),
                one::<T>(),
                par,
            );
            start = end;
        }
        alphabar.copy_from(&alpha);
        pbar_adjoint.fill(zero());
        let mut start = 0;
        while start < k {
            let end = Ord::min(k - start, s) + start;
            let len = end - start;
            pbar_adjoint
                .rb_mut()
                .get_mut(..2 * len, 2 * start..2 * end)
                .diagonal_mut()
                .column_vector_mut()
                .fill(one());
            start = end;
        }
        theta.fill(zero());
        w.fill(zero());
        wbar.fill(zero());
        for iter in 0..max_iters {
            // An exactly zero normal residual leaves no search direction.
            // The callback has already had the opportunity to accept this iterate.
            if zetabar.norm_max() == zero::<T::Real>() {
                return LsmrInfo {
                    iter_count: iter,
                    stop: LsmrStop::Breakdown,
                };
            }
            let (mut vold, mut stack) = unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
            let mut vold = vold.as_mat_mut();
            {
                let (mut qr, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(m, k, stack.rb_mut()) };
                let mut qr = qr.as_mat_mut();
                vold.copy_from(&v);
                M.apply_in_place(v.rb_mut(), par, stack.rb_mut());
                A.apply(qr.rb_mut(), v.rb(), par, stack.rb_mut());
                let mut start = 0;
                while start < k {
                    let s = Ord::min(k - start, s);
                    let end = start + s;
                    matmul(
                        qr.rb_mut().get_mut(.., start..end),
                        Accum::Add,
                        u.rb().get(.., start..end),
                        alpha.rb().get(..s, start..end).adjoint(),
                        -one::<T>(),
                        par,
                    );
                    thin_qr(
                        u.rb_mut().get_mut(.., start..end),
                        beta.rb_mut().get_mut(..s, start..end),
                        qr.rb_mut().get_mut(.., start..end),
                        par,
                        stack.rb_mut(),
                    );
                    start = end;
                }
            }
            {
                let (mut qr, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(n, k, stack.rb_mut()) };
                let mut qr = qr.as_mat_mut();
                A.adjoint_apply(qr.rb_mut(), u.rb(), par, stack.rb_mut());
                M.adjoint_apply_in_place(qr.rb_mut(), par, stack.rb_mut());
                let mut start = 0;
                while start < k {
                    let s = Ord::min(k - start, s);
                    let end = start + s;
                    matmul(
                        qr.rb_mut().get_mut(.., start..end),
                        Accum::Add,
                        vold.rb().get(.., start..end),
                        beta.rb().get(..s, start..end).adjoint(),
                        -one::<T>(),
                        par,
                    );
                    vold.rb_mut()
                        .get_mut(.., start..end)
                        .copy_from(v.rb().get(.., start..end));
                    thin_qr(
                        v.rb_mut().get_mut(.., start..end),
                        alpha.rb_mut().get_mut(..s, start..end),
                        qr.rb_mut().get_mut(.., start..end),
                        par,
                        stack.rb_mut(),
                    );
                    start = end;
                }
            }
            let mut Mvold = vold;
            let mut start = 0;
            while start < k {
                let s = Ord::min(k - start, s);
                let end = start + s;
                let mut x = x.rb_mut().get_mut(.., start..Ord::min(actual_k, end));
                let mut Mvold = Mvold.rb_mut().get_mut(.., start..end);
                let mut w = w.rb_mut().get_mut(.., start..end);
                let mut wbar = wbar.rb_mut().get_mut(.., start..end);
                let alpha = alpha.rb_mut().get_mut(..s, start..end);
                let beta = beta.rb_mut().get_mut(..s, start..end);
                let mut zetabar = zetabar.rb_mut().get_mut(..s, start..end);
                let mut alphabar = alphabar.rb_mut().get_mut(..s, start..end);
                let mut theta = theta.rb_mut().get_mut(..s, start..end);
                let mut pbar_adjoint = pbar_adjoint.rb_mut().get_mut(..2 * s, 2 * start..2 * end);
                let (mut p_adjoint, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(2 * s, 2 * s, stack.rb_mut()) };
                let mut p_adjoint = p_adjoint.as_mat_mut();
                let (mut rho, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
                let mut rho = rho.as_mat_mut();
                let (mut thetaold, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
                let mut thetaold = thetaold.as_mat_mut();
                let (mut rhobar, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
                let mut rhobar = rhobar.as_mat_mut();
                let (mut thetabar, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
                let mut thetabar = thetabar.as_mat_mut();
                let (mut zeta, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
                let mut zeta = zeta.as_mat_mut();
                let (mut zetabar_tmp, mut stack) =
                    unsafe { temp_mat_uninit::<T, _, _>(s, s, stack.rb_mut()) };
                let mut zetabar_tmp = zetabar_tmp.as_mat_mut();
                {
                    let (mut qr, mut stack) =
                        unsafe { temp_mat_uninit::<T, _, _>(2 * s, s, stack.rb_mut()) };
                    let mut qr = qr.as_mat_mut();
                    qr.rb_mut()
                        .get_mut(..s, ..)
                        .copy_from(alphabar.rb().adjoint());
                    qr.rb_mut().get_mut(s.., ..).copy_from(&beta);
                    thin_qr(
                        p_adjoint.rb_mut(),
                        rho.rb_mut(),
                        qr.rb_mut(),
                        par,
                        stack.rb_mut(),
                    );
                }
                thetaold.copy_from(&theta);
                matmul(
                    theta.rb_mut(),
                    Accum::Replace,
                    alpha.rb(),
                    p_adjoint.rb().get(s.., ..s),
                    one::<T>(),
                    par,
                );
                matmul(
                    alphabar.rb_mut(),
                    Accum::Replace,
                    alpha.rb(),
                    p_adjoint.rb().get(s.., s..),
                    one::<T>(),
                    par,
                );
                matmul(
                    thetabar.rb_mut(),
                    Accum::Replace,
                    rho.rb(),
                    pbar_adjoint.rb().get(s.., ..s),
                    one::<T>(),
                    par,
                );
                {
                    let (mut qr, mut stack) =
                        unsafe { temp_mat_uninit::<T, _, _>(2 * s, s, stack.rb_mut()) };
                    let mut qr = qr.as_mat_mut();
                    matmul(
                        qr.rb_mut().get_mut(..s, ..),
                        Accum::Replace,
                        pbar_adjoint.rb().adjoint().get(s.., s..),
                        rho.rb().adjoint(),
                        one::<T>(),
                        par,
                    );
                    qr.rb_mut().get_mut(s.., ..).copy_from(&theta);
                    thin_qr(
                        pbar_adjoint.rb_mut(),
                        rhobar.rb_mut(),
                        qr.rb_mut(),
                        par,
                        stack.rb_mut(),
                    );
                }
                matmul(
                    zeta.rb_mut(),
                    Accum::Replace,
                    pbar_adjoint.rb().adjoint().get(..s, ..s),
                    zetabar.rb(),
                    one::<T>(),
                    par,
                );
                matmul(
                    zetabar_tmp.rb_mut(),
                    Accum::Replace,
                    pbar_adjoint.rb().adjoint().get(s.., ..s),
                    zetabar.rb(),
                    one::<T>(),
                    par,
                );
                zetabar.copy_from(&zetabar_tmp);
                matmul(
                    Mvold.rb_mut(),
                    Accum::Add,
                    w.rb(),
                    thetaold.rb().adjoint(),
                    -one::<T>(),
                    par,
                );
                faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    rho.rb().transpose(),
                    Mvold.rb_mut().transpose_mut(),
                    par,
                );
                w.copy_from(&Mvold);
                matmul(
                    Mvold.rb_mut(),
                    Accum::Add,
                    wbar.rb(),
                    thetabar.rb().adjoint(),
                    -one::<T>(),
                    par,
                );
                faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    rhobar.rb().transpose(),
                    Mvold.rb_mut().transpose_mut(),
                    par,
                );
                wbar.copy_from(&Mvold);
                let actual_s = x.ncols();
                matmul(
                    x.rb_mut(),
                    if iter == 0 {
                        Accum::Replace
                    } else {
                        Accum::Add
                    },
                    wbar.rb(),
                    zeta.rb().get(.., ..actual_s),
                    one::<T>(),
                    par,
                );
                start = end;
            }
            if let ControlFlow::Break(value) = callback(x.rb()) {
                return LsmrInfo {
                    iter_count: iter + 1,
                    stop: LsmrStop::Callback(value),
                };
            }
        }
        LsmrInfo {
            iter_count: max_iters,
            stop: LsmrStop::IterationLimit,
        }
    }
    implementation(
        out,
        &right_precond,
        &mat,
        rhs,
        max_iters,
        &mut { callback },
        par,
        stack,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Mat, dyn_stack::MemBuffer, matrix_free::InitialGuessStatus};

    #[test]
    fn faer_stopping_criterion_matches_faer() {
        let rel_tolerance = 1e-10;
        let a = Mat::from_fn(9, 6, |i, j| {
            if i == j {
                2.0 + i as f64
            } else {
                ((i * 7 + j * 3) as f64).sin()
            }
        });
        let precond = Mat::from_fn(
            6,
            6,
            |i, j| {
                if i == j { 1.0 / (1.0 + i as f64) } else { 0.0 }
            },
        );
        for k in [1, 2, 8] {
            let b = Mat::from_fn(9, k, |i, j| ((i + 5 * j + 1) as f64).cos());
            let mut expected = Mat::from_fn(6, k, |i, j| (i + j) as f64 / 10.0);
            let mut actual = expected.clone();
            let mut buffer =
                MemBuffer::new(lsmr_scratch(precond.as_ref(), a.as_ref(), k, Par::Seq));
            let reference = faer::matrix_free::lsmr::lsmr(
                expected.as_mut(),
                precond.as_ref(),
                a.as_ref(),
                b.as_ref(),
                faer::matrix_free::lsmr::LsmrParams {
                    initial_guess: InitialGuessStatus::Zero,
                    rel_tolerance,
                    max_iters: 100,
                    ..Default::default()
                },
                |_| {},
                Par::Seq,
                MemStack::new(&mut buffer),
            )
            .unwrap();
            // Faer tests the preconditioned normal residual relative to Mᵀ Aᵀ b.
            // Evaluate it explicitly here rather than using its recurrence estimate.
            let normal_operator = precond.transpose() * a.transpose();
            let threshold = rel_tolerance * (&normal_operator * &b).norm_l2();
            let result = lsmr_with_control(
                actual.as_mut(),
                precond.as_ref(),
                a.as_ref(),
                b.as_ref(),
                100,
                |x| {
                    let residual = &b - &a * x;
                    if (&normal_operator * &residual).norm_l2() <= threshold {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                },
                Par::Seq,
                MemStack::new(&mut buffer),
            );
            assert_eq!(result.stop, LsmrStop::Callback(()));
            assert_eq!(result.iter_count, reference.iter_count);
            assert_eq!(actual, expected);
        }
    }
}
