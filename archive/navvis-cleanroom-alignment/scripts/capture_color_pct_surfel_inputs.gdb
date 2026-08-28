set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on

starti

break _ZN6navvis10pointcloud21NormalFieldAggregatorINS0_18PointXYZRGBINormalEE8addPointERKS2_ if *(float*)$rsi >= -1.5682510375976566 && *(float*)$rsi < -1.508251037597656 && *(float*)($rsi + 4) >= 4.410560989379881 && *(float*)($rsi + 4) < 4.470560989379884 && *(float*)($rsi + 8) >= 2.3516017913818352 && *(float*)($rsi + 8) < 2.411601791381834
commands
  silent
  printf "PCT_SURFEL_INPUT xyz=(%.17g %.17g %.17g) normal=(%.17g %.17g %.17g) curvature=%.17g bits=(0x%08x 0x%08x 0x%08x 0x%08x)\n", *(float*)$rsi, *(float*)($rsi + 4), *(float*)($rsi + 8), *(float*)($rsi + 16), *(float*)($rsi + 20), *(float*)($rsi + 24), *(float*)($rsi + 40), *(unsigned int*)($rsi + 16), *(unsigned int*)($rsi + 20), *(unsigned int*)($rsi + 24), *(unsigned int*)($rsi + 40)
  continue
end

break _ZN6navvis10pointcloud21NormalFieldAggregatorINS0_18PointXYZRGBINormalEE8getValueERS2_i if *(unsigned int*)$rsi == 0xbfc51e20 && *(unsigned int*)($rsi + 4) == 0x408d4cea && *(unsigned int*)($rsi + 8) == 0x40184e84
commands
  silent
  printf "PCT_SURFEL_SUM count=%d sum=(%.17g %.17g %.17g) curvature_sum=%.17g mxcsr=0x%x\n", $edx, *(float*)($rdi + 8), *(float*)($rdi + 12), *(float*)($rdi + 16), *(float*)($rdi + 20), $mxcsr
  quit
end

continue
quit
