f90_template="""module solve_ida

   implicit none
   
   integer*8 :: neq = {neq}
   integer*8 :: iout(25)
   real*8 :: rout(10)
   real*8 :: y0({neq}), yp0({neq})
   real*8 :: diff({neq}), mas({neq}, {neq})

end module solve_ida

subroutine initialize(neqin, y0in, rtol, atol, ipar, rpar, id_vec)

   use solve_ida, only: neq, iout, rout, y0, yp0, mas, diff
   
   implicit none
   
   integer*8, intent(in) :: neqin, ipar(*)
   real*8, intent(in) :: y0in(neqin), rtol, atol(*)
   real*8, intent(in) :: rpar(*)
   real*8, intent(in) :: id_vec(neqin)
   real*8 :: constr_vec(neqin)
   real*8 :: t0, yptmp(neqin)
   integer*8 :: nthreads, iatol, ier
   integer*8 :: i
   integer, external :: OMP_GET_NUM_THREADS
   
   iatol = 2
   constr_vec = 1
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
   ! set maximum number of steps
   call fidasetiin('MAX_NSTEPS', 500000, ier)
   ! set algebraic variables
   call fidasetvin('ID_VEC', id_vec, ier)
   ! set constraints (all yi >= 0.)
   call fidasetvin('CONSTR_VEC', constr_vec, ier)
   ! initialize the solver
   call fidadense(neq, ier)
   ! enable the jacobian
   call fidadensesetjac(1, ier)

end subroutine initialize

subroutine solve(neqin, nrates, nt, tfinal, t1, u1, du1, r1)

   use solve_ida, only: y0, yp0
   
   implicit none
   
   integer*8, intent(in) :: neqin, nt, nrates
   real*8, intent(in) :: tfinal
   
   real*8, intent(out) :: t1(nt)
   real*8, intent(out) :: u1(neqin, nt), du1(neqin, nt)
   real*8, intent(out) :: r1(nrates, nt)
   
   real*8 :: dt, tout
   integer*8 :: itask, ier
   integer*8 :: i

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
   
   integer*8, intent(in) :: ipar(*)
   integer*8, intent(out) :: reserr
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
   
   integer*8 :: neqin, ipar(*)
   integer*8 :: djacerr
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
   """


pyf_template = """!    -*- f90 -*-
! Note: the context of this file is case sensitive.

python module {modname} ! in 
    interface  ! in :{modname}
        module solve_ida ! in :{modname}:{modname}.f90
            real*8 dimension({neq}) :: yp0
            real*8 dimension(10) :: rout
            integer*8 dimension(25) :: iout
            real*8 dimension({neq},{neq}) :: mas
            real*8 dimension({neq}) :: y0
            real*8 dimension({neq}) :: diff
            integer*8, optional :: neq={neq}
        end module solve_ida
        subroutine initialize(neqin,y0in,rtol,atol,ipar,rpar,id_vec) ! in :{modname}:{modname}.f90
            use solve_ida, only: mas,rout,iout,yp0,diff,y0,neq
            integer*8, optional,intent(in),check(len(y0in)>=neqin),depend(y0in) :: neqin=len(y0in)
            real*8 dimension(neqin),intent(in) :: y0in
            real*8 intent(in) :: rtol
            real*8 dimension(*),intent(in) :: atol
            integer*8 dimension(*),intent(in) :: ipar
            real*8 dimension(*),intent(in) :: rpar
            real*8 dimension(neqin),intent(in),depend(neqin) :: id_vec
        end subroutine initialize
        subroutine solve(neqin,nrates,nt,tfinal,t1,u1,du1,r1) ! in :{modname}:{modname}.f90
            use solve_ida, only: y0,yp0,neq
            integer*8 intent(in) :: neqin
            integer*8 intent(in) :: nrates
            integer*8 intent(in) :: nt
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
