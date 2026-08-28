set pagination off
set print thread-events off
set confirm off
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_abs_probe/abs_runtime.log
set logging overwrite on
set logging redirect on
set logging enabled on
set $abs_seen = 0
set $abs_vector = 0
starti

# Hash-frozen nv_colorcloud Build ID a7586f518009434f5e97891f897aea42675f26a0.
# This call is the optional view-weight adjustment in the final direct-color
# blender.  RSI points at std::vector<float>; RDI is the adjustment object.
break *0x555555792fce
commands
  silent
  if $abs_seen == 0
    set $abs_seen = 1
    set $abs_vector = $rsi
    set $abs_begin = *(float**)$rsi
    set $abs_end = *(float**)($rsi + 8)
    set $abs_count = $abs_end - $abs_begin
    printf "ABS_OBJECT object=%p vtable=%p target=%p count=%ld\n", $rdi, *(void**)$rdi, *(void**)(*(void**)$rdi + 0x10), $abs_count
    printf "ABS_OBJECT_WORDS "
    x/16gx $rdi
    set $sigma_begin = *(float**)($rdi + 8)
    set $sigma_end = *(float**)($rdi + 16)
    set $sigma_count = $sigma_end - $sigma_begin
    printf "ABS_SIGMA_TABLE count=%ld ", $sigma_count
    set $j = 0
    while $j < $sigma_count
      printf "%.17g/0x%08x ", $sigma_begin[$j], *(unsigned int*)($sigma_begin + $j)
      set $j = $j + 1
    end
    printf "\n"
    printf "ABS_INPUT "
    set $i = 0
    while $i < $abs_count
      printf "%.17g/0x%08x ", $abs_begin[$i], *(unsigned int*)($abs_begin + $i)
      set $i = $i + 1
    end
    printf "\n"
    disable 1
  end
  continue
end

break *0x555555792fd2
commands
  silent
  if $abs_seen == 1
    set $abs_begin = *(float**)$abs_vector
    set $abs_end = *(float**)($abs_vector + 8)
    set $abs_count = $abs_end - $abs_begin
    printf "ABS_OUTPUT "
    set $i = 0
    while $i < $abs_count
      printf "%.17g/0x%08x ", $abs_begin[$i], *(unsigned int*)($abs_begin + $i)
      set $i = $i + 1
    end
    printf "\n"
    quit
  end
  continue
end

continue
