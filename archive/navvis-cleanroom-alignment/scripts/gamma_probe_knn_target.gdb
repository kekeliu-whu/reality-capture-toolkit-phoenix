set pagination off
set print thread-events off
set confirm off
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_knn_replay/vendor_target_2296953.log
set logging overwrite on
set logging redirect on
set logging enabled on
set $armed = 0
set $target_thread = -1
starti

# Exact xyz float bits of point 2296953 in the frozen input geometry.
break *0x5555557a7dc0 if *(unsigned int*)$rsi == 0x3f944066 && *(unsigned int*)($rsi+4) == 0x407276e7 && *(unsigned int*)($rsi+8) == 0x3f83b446
commands
  silent
  set scheduler-locking on
  set $armed = 1
  set $target_thread = $_thread
  enable 2
  printf "TARGET_ENTRY thread=%d query=%p xyz_bits=%08x,%08x,%08x\n", $_thread, $rsi, *(unsigned int*)$rsi, *(unsigned int*)($rsi+4), *(unsigned int*)($rsi+8)
  continue
end

# Return site in the worker calling the nearestKSearch wrapper.
break *0x5555557a6d91
commands
  silent
  if $armed == 1 && $_thread == $target_thread
    set $indices = *(int**)$rbp
    set $distances = *(float**)$r12
    printf "TARGET_RESULT count=%d indices=(%d %d %d %d %d) squared_distances=(%.17g/0x%08x %.17g/0x%08x %.17g/0x%08x %.17g/0x%08x %.17g/0x%08x)\n", $eax, $indices[0], $indices[1], $indices[2], $indices[3], $indices[4], $distances[0], *(unsigned int*)($distances+0), $distances[1], *(unsigned int*)($distances+1), $distances[2], *(unsigned int*)($distances+2), $distances[3], *(unsigned int*)($distances+3), $distances[4], *(unsigned int*)($distances+4)
    quit
  end
  continue
end
disable 2

continue
