set pagination off
set print thread-events off
set confirm off
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_knn_replay/vendor_arithmetic_2336288.log
set logging overwrite on
set logging redirect on
set logging enabled on
set $armed = 0
set $target_thread = -1
starti

# Exact xyz float bits of fallback point 2336288 in the frozen geometry.
break *0x5555557a7dc0 if *(unsigned int*)$rsi == 0x3f6f65cf && *(unsigned int*)($rsi+4) == 0x407d943e && *(unsigned int*)($rsi+8) == 0x3f0e44fe
commands
  silent
  set scheduler-locking on
  set $armed = 1
  set $target_thread = $_thread
  enable 2
  enable 3
  enable 4
  printf "TARGET_ENTRY thread=%d query=%p xyz_bits=%08x,%08x,%08x normal_bits=%08x,%08x,%08x\n", $_thread, $rsi, *(unsigned int*)$rsi, *(unsigned int*)($rsi+4), *(unsigned int*)($rsi+8), *(unsigned int*)($rsi+0x10), *(unsigned int*)($rsi+0x14), *(unsigned int*)($rsi+0x18)
  continue
end

# nearestKSearch has returned; preserve exact compact IDs and distance bits.
break *0x5555557a6d91
commands
  silent
  if $armed == 1 && $_thread == $target_thread
    set $indices = *(int**)$rbp
    set $distances = *(float**)$r12
    printf "NEIGHBORS count=%d indices=(%d %d %d %d %d) distance_bits=(%08x %08x %08x %08x %08x)\n", $eax, $indices[0], $indices[1], $indices[2], $indices[3], $indices[4], *(unsigned int*)($distances+0), *(unsigned int*)($distances+1), *(unsigned int*)($distances+2), *(unsigned int*)($distances+3), *(unsigned int*)($distances+4)
  end
  continue
end
disable 2

# Blend helper entry. Its compact seed cloud uses 48-byte points, and the
# neighbor rows index directly into that compact cloud.
break *0x5555557a70b0
commands
  silent
  if $armed == 1 && $_thread == $target_thread
    set $helper_indices = *(int**)$rsi
    set $helper_distances = *(float**)$rdx
    set $seed_base = *(unsigned char**)$r8
    printf "BLEND_ENTRY count=%d query=%p seed_base=%p\n", $edi, $rcx, $seed_base
    set $rank = 0
    while $rank < $edi
      set $seed = $seed_base + $helper_indices[$rank] * 48
      printf "SEED rank=%d compact=%d distance=%08x xyz=%08x,%08x,%08x rgb=%u,%u,%u normal=%08x,%08x,%08x\n", $rank, $helper_indices[$rank], *(unsigned int*)($helper_distances+$rank), *(unsigned int*)$seed, *(unsigned int*)($seed+4), *(unsigned int*)($seed+8), *(unsigned char*)($seed+0x24), *(unsigned char*)($seed+0x25), *(unsigned char*)($seed+0x26), *(unsigned int*)($seed+0x10), *(unsigned int*)($seed+0x14), *(unsigned int*)($seed+0x18)
      set $rank = $rank + 1
    end
    disable 3
  end
  continue
end
disable 3

# Return point of the blend helper, before its stack frame is released.
# The SIMD accumulator is [blue_sum, green_sum, red_sum, total_weight].
break *0x5555557a73aa
commands
  silent
  if $armed == 1 && $_thread == $target_thread
    printf "BLEND_RETURN accum_bits=(%08x %08x %08x %08x) accum=(%.17g %.17g %.17g %.17g)\n", *(unsigned int*)($rsp+0x10), *(unsigned int*)($rsp+0x14), *(unsigned int*)($rsp+0x18), *(unsigned int*)($rsp+0x1c), *(float*)($rsp+0x10), *(float*)($rsp+0x14), *(float*)($rsp+0x18), *(float*)($rsp+0x1c)
    quit
  end
  continue
end
disable 4

continue
