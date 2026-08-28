set pagination off
set confirm off
starti
break *0x5555557c6770
python
import os
skip = int(os.environ.get("NAVVIS_IMU_FACTOR_SKIP", "0"))
if skip:
    gdb.execute(f"ignore 1 {skip}")
end
continue
set $residuals = $rdx
set $parameters = $rsi
printf "EXACT_ENTRY this=%p residuals=%p num_residuals=%d\n", $rdi, $residuals, *(int*)($rdi+32)
printf "FUNCTOR_OBJECT\n"
set $functor = *(void**)($rdi+40)
x/16gx $functor
set $functor_p0 = *(void**)($functor+0)
set $functor_p1 = *(void**)($functor+8)
set $functor_p2 = *(void**)($functor+16)
set $functor_p3 = *(void**)($functor+24)
set $functor_p4 = *(void**)($functor+32)
set $functor_p5 = *(void**)($functor+40)
set $functor_p6 = *(void**)($functor+48)
set $functor_p7 = *(void**)($functor+56)
printf "FUNCTOR_POINTER_0 %p\n", $functor_p0
x/32gx $functor_p0
printf "FUNCTOR_POINTER_1 %p\n", $functor_p1
x/32gx $functor_p1
printf "FUNCTOR_POINTER_2 %p\n", $functor_p2
x/32gx $functor_p2
printf "FUNCTOR_POINTER_3 %p\n", $functor_p3
x/32gx $functor_p3
printf "FUNCTOR_POINTER_4 %p\n", $functor_p4
x/32gx $functor_p4
printf "FUNCTOR_POINTER_5 %p\n", $functor_p5
x/32gx $functor_p5
printf "FUNCTOR_POINTER_6 %p\n", $functor_p6
x/32gx $functor_p6
printf "FUNCTOR_POINTER_7 %p\n", $functor_p7
x/32gx $functor_p7
printf "P00_IMU_ORIENTATION\n"
x/4gf *(void**)($parameters+0)
printf "P01_IMU_TRANSLATION\n"
x/3gf *(void**)($parameters+8)
printf "P02_SOURCE_ROTATION\n"
x/4gf *(void**)($parameters+16)
printf "P03_SOURCE_TRANSLATION\n"
x/3gf *(void**)($parameters+24)
printf "P04_TARGET_ROTATION\n"
x/4gf *(void**)($parameters+32)
printf "P05_TARGET_TRANSLATION\n"
x/3gf *(void**)($parameters+40)
printf "P06_SOURCE_VELOCITY\n"
x/3gf *(void**)($parameters+48)
printf "P07_TARGET_VELOCITY\n"
x/3gf *(void**)($parameters+56)
printf "P08_ACCEL_BIAS\n"
x/3gf *(void**)($parameters+64)
printf "P09_ACCEL_SCALE\n"
x/3gf *(void**)($parameters+72)
printf "P10_ACCEL_CROSS_AXIS\n"
x/6gf *(void**)($parameters+80)
printf "P11_GYRO_BIAS\n"
x/3gf *(void**)($parameters+88)
printf "P12_GYRO_SCALE\n"
x/3gf *(void**)($parameters+96)
printf "P13_GYRO_CROSS_AXIS\n"
x/6gf *(void**)($parameters+104)
printf "P14_GRAVITY\n"
x/1gf *(void**)($parameters+112)
disable 1
finish
printf "EXACT_RESIDUALS\n"
x/9gf $residuals
quit
