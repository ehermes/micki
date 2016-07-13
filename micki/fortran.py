f90_template="""module solve_ida

   implicit none

   integer :: neq = {neq}
   integer :: iout(25)
   real*8 :: rout(10)
   real*8 :: y0({neq}), yp0({neq})
   real*8 :: diff({neq}), mas({neq}, {neq})
   real*8 :: jac({neq}, {neq})

end module solve_ida

subroutine initialize(neqin, y0in, rtol, atol, ipar, rpar, id_vec)

   use solve_ida, only: neq, iout, rout, y0, yp0, mas, diff

   implicit none

   integer, intent(in) :: neqin, ipar(*)
   real*8, intent(in) :: y0in(neqin), rtol, atol(*)
   real*8, intent(in) :: rpar(*)
   real*8, intent(in) :: id_vec(neqin)
   real*8 :: constr_vec(neqin)
   real*8 :: t0, yptmp(neqin)
   integer :: nthreads, iatol, ier
   integer :: i
   integer, external :: OMP_GET_NUM_THREADS

   iatol = 2
   constr_vec = 1.d0
   nthreads = OMP_GET_NUM_THREADS()

   y0 = y0in
   yp0 = 0
   yptmp = 0
   diff = id_vec
   mas = 0
   t0 = 0

   do i = 1, neq
      mas(i, i) = id_vec(i)
   enddo

   ! Calculate yp
   call fidaresfun(0.d0, y0, yptmp, yp0, ipar, rpar, ier)
   ! initialize Sundials
   call fnvinits(2, neq, ier)
   !call fnvinitomp(2, neq, nthreads, ier)
   ! allocate memory
   call fidamalloc(t0, y0, yp0, iatol, rtol, atol, iout, &
                   rout, ipar, rpar, ier)
   ! set maximum number of steps (default = 500)
   call fidasetiin('MAX_NSTEPS', 500000, ier)
!   ! set maximum number of nonlinear iterations (default = 4)
!   call fidasetiin('MAX_NITERS', 2, ier)
!   ! set maximum number of error test failures (default = 10)
!   call fidasetiin('MAX_ERRFAILS', 50, ier)
!   ! set maximum number of convergence failures (default = 10)
!   call fidasetiin('MAX_CONVFAIL', 50, ier)
   ! set maximum order for LMM method (default = 5)
   call fidasetiin('MAX_ORD', 2, ier)
!   ! set nonlinear convergence test coefficient (default = 0.33)
!   call fidasetrin('NLCONV_COEF', 0.67d0, ier)
   ! set initial step size
   call fidasetrin('INIT_STEP', 1d-17, ier)
   ! set algebraic variables
   call fidasetvin('ID_VEC', id_vec, ier)
   ! set constraints (all yi >= 0.)
   call fidasetvin('CONSTR_VEC', constr_vec, ier)
   ! initialize the solver
   call fidalapackdense(neq, ier)
   ! enable the jacobian
   call fidalapackdensesetjac(1, ier)
!   ! initialize the solver
!!   call fidaspgmr(0, 0, 0, 0, 0, ier)
!!   call fidaspbcg(0, 0, 0, ier)
!   call fidasptfqmr(0, 0, 0, ier)
!   ! enable the jacobian
!   call fidaspilssetjac(1, ier)
!   ! enable the preconditioner
!   call fidaspilssetprec(1, ier)

end subroutine initialize

subroutine solve(neqin, nrates, nt, tfinal, t1, u1, du1, r1)

   use solve_ida, only: y0, yp0, iout, rout

   implicit none

   integer, intent(in) :: neqin, nt, nrates
   real*8, intent(in) :: tfinal

   real*8 :: yp(neqin)
   real*8 :: rpar(1)
   integer :: ipar(1)

   real*8, intent(out) :: t1(nt)
   real*8, intent(out) :: u1(neqin, nt), du1(neqin, nt)
   real*8, intent(out) :: r1(nrates, nt)

   real*8 :: dt, tout
   integer :: itask, ier
   integer :: i

   yp = 0.d0

   itask = 1
   dt = tfinal / (nt - 1)
   tout = 0.0d0
   u1 = 0
   du1 = 0
   u1(:, 1) = y0
   du1(:, 1) = yp0
   t1(1) = 0.d0
   call ratecalc({neq}, {nrates}, u1(:, 1), r1(:, 1))

   do i = 2, nt
      tout = tout + dt
      call fidasolve(tout, t1(i), u1(:, i), du1(:, i), itask, ier)
!      call fidaresfun(t1(i), u1(:, i), yp, du1(:, i), ipar, rpar, ier)
      call ratecalc({neq}, {nrates}, u1(:, i), r1(:, i))
   end do

end subroutine solve

subroutine finalize

!   use solve_ida, only: y0, yp0, jac, mas, diff

   implicit none

   call fidafree

end subroutine finalize

subroutine fidaresfun(tres, yin, ypin, res, ipar, rpar, reserr)

   use solve_ida, only: neq, diff

   implicit none

   integer, intent(in) :: ipar(*)
   integer, intent(out) :: reserr
   real*8, intent(in) :: tres, rpar(*)
   real*8, intent(in) :: yin(neq), ypin(neq)
   real*8, intent(out) :: res(neq)
   real*8 :: x({nx})

   res = 0

{rescalc}

   res = res - diff * ypin
   reserr = 0

end subroutine fidaresfun

subroutine fidadjac(neqin, t, yin, ypin, r, jac, cj, ewt, h, &
                    ipar, rpar, wk1, wk2, wk3, djacerr)

use solve_ida, only: mas

   implicit none

   integer :: neqin, ipar(*)
   integer :: djacerr
   real*8 :: t, h, cj, rpar(*)
   real*8 :: yin(neqin), ypin(neqin), r(neqin), ewt(*), jac(neqin, neqin)
   real*8 :: wk1(*), wk2(*), wk3(*)

   jac = 0

{jaccalc}

   jac = jac - cj * mas
   djacerr = 0

end subroutine fidadjac

subroutine ratecalc(neqin, nrates, yin, rates)

   implicit none

   integer, intent(in) :: neqin, nrates
   real*8, intent(in) :: yin(neqin)
   real*8, intent(out) :: rates(nrates)

   rates = 0

{ratecalc}

end subroutine ratecalc

subroutine fidajtimes(tres, yin, ypin, res, vin, fjv, cj, ewt, h, ipar, rpar, wk1, wk2, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), ypin(neq), res(neq), vin(neq), cj, h
   real*8 :: ewt(*), wk1(*), wk2(*), rpar(*)
   integer :: ipar(*)
   integer :: i

   real*8, intent(out) :: fjv(neq)
   integer, intent(out) :: ier

!   real*8 :: jac(neq, neq)
!
!   call fidadjac(neq, tres, yin, ypin, res, jac, cj, ewt, h, &
!                    ipar, rpar, wk1, wk2, wk2, ier)

   do i = 1, neq
   fjv(i) = dot_product(vin, jac(i, :))
   enddo

   ier = 0

end subroutine fidajtimes

subroutine fidapsol(tres, yin, ypin, res, rvin, zv, cj, delta, ewt, ipar, rpar, wk1, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), ypin(neq), res(neq), rvin(neq)
   real*8, intent(in) :: cj, delta, ewt(*), rpar(*)
   integer, intent(in) :: ipar(*)

   real*8 :: wk1(*)
   integer :: ier

   real*8, intent(out) :: zv(neq)

   real*8 :: jaclu(neq, neq)
   integer :: ipiv(neq)
   integer :: i

   jaclu = jac
!   call fidadjac(neq, tres, yin, ypin, res, jaclu, cj, ewt, 1, &
!                    ipar, rpar, wk1, wk1, wk1, ier)

   zv = rvin
   call dgesv(neq, 1, jaclu, neq, ipiv, zv, neq, ier)

end subroutine fidapsol

subroutine fidapset(tres, yin, ypin, res, cj, ewt, h, ipar, rpar, wk1, wk2, wk3, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), ypin(neq), res(neq)
   real*8, intent(in) :: cj, ewt(*), h, rpar(*)
   real*8 :: wk1(*), wk2(*), wk3(*)

   integer, intent(in) :: ipar(*)

   integer, intent(out) :: ier

   call fidadjac(neq, tres, yin, ypin, res, jac, cj, ewt, h, &
                    ipar, rpar, wk1, wk2, wk3, ier)

end subroutine fidapset

   """


pyf_template = """!    -*- f90 -*-
! Note: the context of this file is case sensitive.

python module {modname} ! in 
    interface  ! in :{modname}
        module solve_ida ! in :{modname}:{modname}.f90
            real*8 dimension({neq}) :: yp0
            real*8 dimension(10) :: rout
            integer dimension(25) :: iout
            real*8 dimension({neq},{neq}) :: mas
            real*8 dimension({neq}) :: y0
            real*8 dimension({neq}) :: diff
            integer, optional :: neq={neq}
        end module solve_ida
        subroutine initialize(neqin,y0in,rtol,atol,ipar,rpar,id_vec) ! in :{modname}:{modname}.f90
            use solve_ida, only: mas,rout,iout,yp0,diff,y0,neq
            integer, optional,intent(in),check(len(y0in)>=neqin),depend(y0in) :: neqin=len(y0in)
            real*8 dimension(neqin),intent(in) :: y0in
            real*8 intent(in) :: rtol
            real*8 dimension(*),intent(in) :: atol
            integer dimension(*),intent(in) :: ipar
            real*8 dimension(*),intent(in) :: rpar
            real*8 dimension(neqin),intent(in),depend(neqin) :: id_vec
        end subroutine initialize
        subroutine solve(neqin,nrates,nt,tfinal,t1,u1,du1,r1) ! in :{modname}:{modname}.f90
            use solve_ida, only: y0,yp0,neq
            integer intent(in) :: neqin
            integer intent(in) :: nrates
            integer intent(in) :: nt
            real*8 intent(in) :: tfinal
            real*8 intent(out),dimension(nt),depend(nt) :: t1
            real*8 intent(out),dimension(neqin,nt),depend(neqin,nt) :: u1
            real*8 intent(out),dimension(neqin,nt),depend(neqin,nt) :: du1
            real*8 intent(out),dimension(nrates,nt),depend(nrates,nt) :: r1
        end subroutine solve
        subroutine finalize ! in :{modname}:{modname}.f90
            use solve_ida, only: mas,y0,yp0,diff
        end subroutine finalize
    end interface 
end python module {modname}

! This file was auto-generated with f2py (version:2).
! See http://cens.ioc.ee/projects/f2py2e/"""
