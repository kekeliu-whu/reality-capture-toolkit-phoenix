set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on

starti

break _ZN6navvis10pointcloud21NormalFieldAggregatorINS0_18PointXYZRGBINormalEE8getValueERS2_i if *(unsigned int*)$rsi == 0xbfc51e20 && *(unsigned int*)($rsi + 4) == 0x408d4cea && *(unsigned int*)($rsi + 8) == 0x40184e84
commands
  silent
  printf "PCT_SURFEL_NORMAL_INPUT output=%p count=%d sum=(%.17g %.17g %.17g) curvature_sum=%.17g\n", $rsi, $edx, *(float*)($rdi + 8), *(float*)($rdi + 12), *(float*)($rdi + 16), *(float*)($rdi + 20)
  set $target_output = $rsi
  finish
  printf "PCT_SURFEL_NORMAL_OUTPUT normal=(%.17g %.17g %.17g) curvature=%.17g bits=(0x%08x 0x%08x 0x%08x)\n", *(float*)($target_output + 16), *(float*)($target_output + 20), *(float*)($target_output + 24), *(float*)($target_output + 40), *(unsigned int*)($target_output + 16), *(unsigned int*)($target_output + 20), *(unsigned int*)($target_output + 24)
  quit
end

continue
quit
