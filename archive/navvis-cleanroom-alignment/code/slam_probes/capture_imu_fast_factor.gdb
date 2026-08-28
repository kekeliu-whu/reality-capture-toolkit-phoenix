set pagination off
set confirm off
starti

# PIE base is 0x555555554000 with GDB disable-randomization. 0x268cb0 is the
# ImuCostFunctionFast DynamicAutoDiffCostFunction::Evaluate vtable entry.
break *0x5555557bccb0
python
import os
skip = int(os.environ.get("NAVVIS_IMU_FAST_FACTOR_SKIP", "0"))
if skip:
    gdb.execute(f"ignore 1 {skip}")
end
continue

set $residuals = $rdx
set $parameters = $rsi
set $sizes = *(int**)($rdi+8)
set $size_end = *(int**)($rdi+16)
set $block_count = $size_end - $sizes
set $residual_count = *(int*)($rdi+32)
printf "FAST_ENTRY this=%p residuals=%p residual_count=%d block_count=%ld\n", $rdi, $residuals, $residual_count, $block_count
printf "FUNCTOR_OBJECT\n"
x/96gx *(void**)($rdi+40)
python
count = int(gdb.parse_and_eval("$block_count"))
sizes = gdb.parse_and_eval("$sizes")
parameters = gdb.parse_and_eval("$parameters")
for index in range(count):
    size = int((sizes + index).dereference())
    pointer = (parameters + index).dereference()
    gdb.write(f"P{index:02d} size={size} address={pointer}\n")
    gdb.execute(f"x/{size}gf {pointer}")
end

disable 1
finish
printf "FAST_RESIDUALS\n"
python
count = int(gdb.parse_and_eval("$residual_count"))
pointer = gdb.parse_and_eval("$residuals")
gdb.execute(f"x/{count}gf {pointer}")
end
quit
