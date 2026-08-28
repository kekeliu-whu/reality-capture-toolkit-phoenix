set pagination off
set confirm off
set disable-randomization on
set environment OMP_NUM_THREADS 1

# Ubuntu's PIE loader uses this base when ASLR is disabled.  The offsets below
# are pinned to build-id 24ef8bee2e35486ec1e9922dd5459cd028bbfd20.
starti
set $image_base = 0x555555554000
set $ray_count = 0
set $hit_count = 0

break *($image_base + 0x1391d0)
commands
  silent
  printf "RENDER this=%p parameterization=%p pose=%p max_distance=%.17g\n", $rdi, $rsi, $rdx, $xmm0.v2_double[0]
  printf "PARAMETERIZATION\n"
  x/8gx $rsi
  printf "POSE\n"
  x/16gx $rdx
  disable 1
  continue
end

break *($image_base + 0x138d20)
commands
  silent
  set $ray_count = $ray_count + 1
  printf "RAY n=%d origin=(%.9g %.9g %.9g) direction=(%.9g %.9g %.9g) max_distance=%.17g\n", $ray_count, *(float*)$rsi, *(float*)($rsi+4), *(float*)($rsi+8), *(float*)$rdx, *(float*)($rdx+4), *(float*)($rdx+8), $xmm0.v2_double[0]
  x/3wx $rsi
  x/3wx $rdx
  if $ray_count >= 8
    disable 2
  end
  continue
end

# Return site immediately after the virtual point-cloud ray query.
break *($image_base + 0x139399)
commands
  silent
  set $hit_count = $hit_count + 1
  printf "QUERY_RETURN n=%d point_index=%d distance=%.17g\n", $hit_count, $eax, $xmm0.v2_double[0]
  if $hit_count >= 8
    disable 3
  end
  continue
end

continue
