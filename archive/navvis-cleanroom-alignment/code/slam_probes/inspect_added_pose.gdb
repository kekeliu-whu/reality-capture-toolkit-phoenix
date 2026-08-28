set pagination off
set confirm off
starti
set $image_base = 0x555555554000
break *($image_base + 0x61de7b)
commands
  silent
  printf "ADDED_POSE_BREAK\n"
  bt 18
  info registers rdi rsi rdx rcx r8 r9 r10 r11 r12 r13 r14 r15 rbp rsp
  x/48gx $rsp
  quit
end
continue
