set pagination off
set confirm off
starti

# PIE base is stable under GDB's default disable-randomization setting.  These
# two addresses are the virtual Evaluate entries recovered from the Fast and
# Exact DynamicAutoDiffCostFunction vtables of build
# 67b4a8b2a22cd09e2b22a9036579e1f4c6a66ea3.
break *0x5555557bccb0
commands
  silent
  printf "FAST_EVALUATE this=%p vptr=%p num_residuals=%d\n", $rdi, *(void**)$rdi, *(int*)($rdi+32)
  printf "PARAM_SIZES begin=%p end=%p cap=%p\n", *(void**)($rdi+8), *(void**)($rdi+16), *(void**)($rdi+24)
  set $begin = *(int**)($rdi+8)
  set $end = *(int**)($rdi+16)
  set $count = $end - $begin
  printf "PARAM_BLOCK_COUNT=%ld\n", $count
  x/24dw $begin
  printf "COST_OBJECT\n"
  x/40gx $rdi
  printf "FUNCTOR_OBJECT\n"
  x/80gx *(void**)($rdi+40)
  printf "P00\n"
  x/4gf *(void**)($rsi+0)
  printf "P01\n"
  x/3gf *(void**)($rsi+8)
  printf "P02\n"
  x/4gf *(void**)($rsi+16)
  printf "P03\n"
  x/3gf *(void**)($rsi+24)
  printf "P04\n"
  x/4gf *(void**)($rsi+32)
  printf "P05\n"
  x/3gf *(void**)($rsi+40)
  printf "P06\n"
  x/3gf *(void**)($rsi+48)
  printf "P07\n"
  x/3gf *(void**)($rsi+56)
  printf "P08\n"
  x/3gf *(void**)($rsi+64)
  printf "P09\n"
  x/3gf *(void**)($rsi+72)
  printf "P10\n"
  x/6gf *(void**)($rsi+80)
  printf "P11\n"
  x/3gf *(void**)($rsi+88)
  printf "P12\n"
  x/3gf *(void**)($rsi+96)
  printf "P13\n"
  x/6gf *(void**)($rsi+104)
  printf "P14\n"
  x/1gf *(void**)($rsi+112)
  disable 1
  continue
end

break *0x5555557c6770
commands
  silent
  printf "EXACT_EVALUATE this=%p vptr=%p num_residuals=%d\n", $rdi, *(void**)$rdi, *(int*)($rdi+32)
  printf "PARAM_SIZES begin=%p end=%p cap=%p\n", *(void**)($rdi+8), *(void**)($rdi+16), *(void**)($rdi+24)
  set $begin = *(int**)($rdi+8)
  set $end = *(int**)($rdi+16)
  set $count = $end - $begin
  printf "PARAM_BLOCK_COUNT=%ld\n", $count
  x/24dw $begin
  printf "COST_OBJECT\n"
  x/40gx $rdi
  printf "FUNCTOR_OBJECT\n"
  x/80gx *(void**)($rdi+40)
  printf "P00\n"
  x/4gf *(void**)($rsi+0)
  printf "P01\n"
  x/3gf *(void**)($rsi+8)
  printf "P02\n"
  x/4gf *(void**)($rsi+16)
  printf "P03\n"
  x/3gf *(void**)($rsi+24)
  printf "P04\n"
  x/4gf *(void**)($rsi+32)
  printf "P05\n"
  x/3gf *(void**)($rsi+40)
  printf "P06\n"
  x/3gf *(void**)($rsi+48)
  printf "P07\n"
  x/3gf *(void**)($rsi+56)
  printf "P08\n"
  x/3gf *(void**)($rsi+64)
  printf "P09\n"
  x/3gf *(void**)($rsi+72)
  printf "P10\n"
  x/6gf *(void**)($rsi+80)
  printf "P11\n"
  x/3gf *(void**)($rsi+88)
  printf "P12\n"
  x/3gf *(void**)($rsi+96)
  printf "P13\n"
  x/6gf *(void**)($rsi+104)
  printf "P14\n"
  x/1gf *(void**)($rsi+112)
  disable 2
  continue
end

continue
