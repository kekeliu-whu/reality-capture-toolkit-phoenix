set pagination off
set confirm off
starti

# AutoDiffCostFunction<AccelerationCostFunction, 3, 4,3,3,3,1,4>::Evaluate.
break *0x5555557e2250
commands
  silent
  set $acc_residuals = $rdx
  set $acc_parameters = $rsi
  printf "ACCELERATION_ENTRY this=%p residuals=%p residual_count=%d\n", $rdi, $acc_residuals, *(int*)($rdi+32)
  printf "ACCELERATION_FUNCTOR\n"
  x/32gx *(void**)($rdi+40)
  printf "A00_ROTATION\n"
  x/4gf *(void**)($acc_parameters+0)
  printf "A01_VELOCITY_FIRST\n"
  x/3gf *(void**)($acc_parameters+8)
  printf "A02_VELOCITY_SECOND\n"
  x/3gf *(void**)($acc_parameters+16)
  printf "A03_GRAVITY_DIRECTION\n"
  x/3gf *(void**)($acc_parameters+24)
  printf "A04_GRAVITY_MAGNITUDE\n"
  x/1gf *(void**)($acc_parameters+32)
  printf "A05_IMU_ORIENTATION\n"
  x/4gf *(void**)($acc_parameters+40)
  disable 1
  finish
  printf "ACCELERATION_RESIDUALS\n"
  x/3gf $acc_residuals
  continue
end

# AutoDiffCostFunction<DeltaRotationCostFunction, 3, 4,4,4>::Evaluate.
break *0x5555557ddfb0
commands
  silent
  set $rotation_residuals = $rdx
  set $rotation_parameters = $rsi
  printf "DELTA_ROTATION_ENTRY this=%p residuals=%p residual_count=%d\n", $rdi, $rotation_residuals, *(int*)($rdi+32)
  printf "DELTA_ROTATION_FUNCTOR\n"
  x/24gx *(void**)($rdi+40)
  printf "R00_ROTATION_FIRST\n"
  x/4gf *(void**)($rotation_parameters+0)
  printf "R01_ROTATION_SECOND\n"
  x/4gf *(void**)($rotation_parameters+8)
  printf "R02_IMU_ORIENTATION\n"
  x/4gf *(void**)($rotation_parameters+16)
  disable 2
  finish
  printf "DELTA_ROTATION_RESIDUALS\n"
  x/3gf $rotation_residuals
  quit
end

continue
