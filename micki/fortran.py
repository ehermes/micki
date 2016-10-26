f90_template = """module solve_ida

   implicit none

   integer :: neq = {neq}
   integer :: iout(21)
   real*8 :: rout(6)
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
   integer :: meth, itmeth
   integer, external :: OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
   integer :: myid

   iatol = 2
   constr_vec = 1.d0

   !$OMP PARALLEL DEFAULT(private) SHARED(nthreads)
   myid = OMP_GET_THREAD_NUM()
   if (myid == 0) then
      nthreads = OMP_GET_NUM_THREADS()
   end if
   !$OMP END PARALLEL

   y0 = y0in
   yp0 = 0
   yptmp = 0
   diff = id_vec
   mas = 0
   t0 = 0
   meth = 2  ! 1 = Adams (nonstiff), 2 = BDF (stiff)
   itmeth = 2  ! 1 = functional iteration, 2 = Newton iteration

!   do i = 1, neq
!      mas(i, i) = id_vec(i)
!   enddo

!   ! Calculate yp
!   call fcvfun(0.d0, y0, yp0, ipar, rpar, ier)

   ! initialize Sundials
   !call fnvinits(1, neq, ier)
   call fnvinitomp(1, neq, nthreads, ier)
   ! allocate memory
   call fcvmalloc(t0, y0, meth, itmeth, iatol, rtol, atol, iout, rout, &
                  ipar, rpar, ier)
! TODO: Fix all of these settings
   ! set maximum number of steps (default = 500)
   call fcvsetiin('MAX_NSTEPS', 50000, ier)
!   ! set maximum number of nonlinear iterations (default = 4)
!!   call fcvsetiin('MAX_NITERS', 2, ier)
!!   ! set maximum number of error test failures (default = 10)
!!   call fcvsetiin('MAX_ERRFAILS', 50, ier)
!!   ! set maximum number of convergence failures (default = 10)
!!   call fcvsetiin('MAX_CONVFAIL', 50, ier)
!   ! set maximum order for LMM method (default = 5)
!   call fcvsetiin('MAX_ORD', 4, ier)
!!   ! set nonlinear convergence test coefficient (default = 0.33)
!!   call fcvsetrin('NLCONV_COEF', 0.67d0, ier)
!   ! set initial step size
!   call fcvsetrin('INIT_STEP', 1d-17, ier)
!   ! set algebraic variables
!   call fcvsetvin('ID_VEC', id_vec, ier)
!   ! set constraints (all yi >= 0.)
!   call fcvsetvin('CONSTR_VEC', constr_vec, ier)
!   ! enable stability limit detection
!   call fcvsetiin('STAB_LIM', 1, ier)
   ! initialize the solver
   call fcvdense(neq, ier)
   ! enable the jacobian
   call fcvdensesetjac(1, ier)
!   ! initialize the solver
!!   call fcvspgmr(2, 2, 0, 0, ier)
!!   call fcvspbcg(0, 0, 0, ier)
!   call fcvsptfqmr(0, 0, 0, ier)
!   ! enable the jacobian
!   call fcvspilssetjac(1, ier)
!   ! enable the preconditioner
!   call fcvspilssetprec(1, ier)

end subroutine initialize

subroutine solve(neqin, nrates, nt, tfinal, t1, u1, du1, r1)

   use solve_ida, only: y0, yp0, iout, rout

   implicit none

   integer, intent(in) :: neqin, nt, nrates
   real*8, intent(in) :: tfinal

   real*8 :: rpar(1)
   integer :: ipar(1)

   real*8, intent(out) :: t1(nt)
   real*8, intent(out) :: u1(neqin, nt), du1(neqin, nt)
   real*8, intent(out) :: r1(nrates, nt)

   real*8 :: dt, tout
   integer :: itask, ier
   integer :: i

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
      do while (tout - t1(i) > dt * 0.01)
         call fcvode(tout, t1(i), u1(:, i), itask, ier)
!         call fcvsolve(tout, t1(i), u1(:, i), du1(:, i), itask, ier)
!         print *, "Target time:", tout
!         print *, "Actual time:", t1(i)
      end do
      call fcvfun(t1(i), u1(:, i), du1(:, i), ipar, rpar, ier)
      call ratecalc({neq}, {nrates}, u1(:, i), r1(:, i))
   end do

end subroutine solve

subroutine finalize

!   use solve_ida, only: y0, yp0, jac, mas, diff

   implicit none

   call fcvfree

end subroutine finalize

subroutine fcvfun(tres, yin, res, ipar, rpar, reserr)

   use solve_ida, only: neq

   implicit none

   integer, intent(in) :: ipar(*)
   integer, intent(out) :: reserr
   real*8, intent(in) :: tres, rpar(*)
   real*8, intent(in) :: yin(neq)
   real*8, intent(out) :: res(neq)
   real*8 :: x({nx})

   res = 0

{rescalc}

   reserr = 0

end subroutine fcvfun

subroutine fcvdjac(neqin, t, yin, ypin, jac, h, &
                    ipar, rpar, wk1, wk2, wk3, djacerr)

   implicit none

   integer :: neqin, ipar(*)
   integer :: djacerr
   real*8 :: t, h, rpar(*)
   real*8 :: yin(neqin), ypin(neqin), jac(neqin, neqin)
   real*8 :: wk1(*), wk2(*), wk3(*)

   jac = 0

{jaccalc}

   djacerr = 0

end subroutine fcvdjac

subroutine ratecalc(neqin, nrates, yin, rates)

   implicit none

   integer, intent(in) :: neqin, nrates
   real*8, intent(in) :: yin(neqin)
   real*8, intent(out) :: rates(nrates)

   rates = 0

{ratecalc}

end subroutine ratecalc

subroutine fcvjtimes(vin, fjv, tres, yin, res, h, ipar, rpar, wk1, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), res(neq), vin(neq), h
   real*8 :: wk1(*), rpar(*)
   integer :: ipar(*)
   integer :: i

   real*8, intent(out) :: fjv(neq)
   integer, intent(out) :: ier

!   real*8 :: jac(neq, neq)
!
!   call fcvdjac(neq, tres, yin, ypin, res, jac, cj, ewt, h, &
!                    ipar, rpar, wk1, wk2, wk2, ier)

   do i = 1, neq
   fjv(i) = dot_product(vin, jac(i, :))
   enddo

   ier = 0

end subroutine fcvjtimes

subroutine fcvpsol(tres, yin, res, rvin, zv, cj, delta, lr, ipar, rpar, wk1, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), res(neq), rvin(neq)
   real*8, intent(in) :: cj, delta, rpar(*)
   integer, intent(in) :: ipar(*), lr

   real*8 :: wk1(*)
   integer :: ier

   real*8, intent(out) :: zv(neq)

   real*8 :: precon(neq, neq)
   integer :: ipiv(neq)
   integer :: i

   precon = -cj * jac
   do i = 1, neq
      precon(i, i) = precon(i, i) + 1
   end do
!   call fcvdjac(neq, tres, yin, ypin, res, jaclu, cj, ewt, 1, &
!                    ipar, rpar, wk1, wk1, wk1, ier)

   zv = rvin
   call dgesv(neq, 1, precon, neq, ipiv, zv, neq, ier)

end subroutine fcvpsol

subroutine fcvpset(tres, yin, res, jok, jcur, cj, h, ipar, rpar, wk1, wk2, wk3, ier)

   use solve_ida, only: neq, jac

   implicit none

   real*8, intent(in) :: tres, yin(neq), res(neq)
   real*8, intent(in) :: cj, h, rpar(*)
   real*8 :: wk1(*), wk2(*), wk3(*)

   integer, intent(in) :: ipar(*), jok

   integer, intent(out) :: ier, jcur

   jcur = 1
   if (jok == 0) then
      call fcvdjac(neq, tres, yin, res, jac, h, &
                   ipar, rpar, wk1, wk2, wk3, ier)
      jcur = 0
   end if

end subroutine fcvpset

   """


pyf_template = """!    -*- f90 -*-
! Note: the context of this file is case sensitive.

python module {modname} ! in
    interface  ! in :{modname}
        module solve_ida ! in :{modname}:{modname}.f90
            real*8 dimension({neq}) :: yp0
            real*8 dimension(6) :: rout
            integer dimension(21) :: iout
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
