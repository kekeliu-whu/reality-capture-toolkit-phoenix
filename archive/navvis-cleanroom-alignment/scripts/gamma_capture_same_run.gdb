set pagination off
set print thread-events off
set confirm off
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_same_run/vendor_capture.log
set logging overwrite on
set logging redirect on
set logging enabled on
set $final_ovs_begin = 0
set $final_ovs_count = 2857623
starti

# Capture the solved model map but continue so that the PLY written by this
# process is guaranteed to use these exact parameters.
break *0x55555573936b
commands
  silent
  set $models = ((void* (*)())0x55555574b450)()
  call ((void (*)(void*, int))0x55555574c480)($models, 1)
  set $node = *(void**)($models + 0x10)
  while $node
    set $view = *(unsigned int*)($node + 8)
    set $gamma = *(void**)($node + 0x10)
    printf "MODEL %u %.17g %.17g\n", $view, *(double*)$gamma, *(double*)($gamma + 8)
    set $node = *(void**)$node
  end
  disable 1
  continue
end

# Locate the final packed 40-byte OVS array.
break *0x555555725a83
commands
  silent
  if $rbp == $final_ovs_count
    set $final_ovs_begin = (unsigned char*)$r12
    printf "FINAL_OVS_SELECTED begin=%p count=%ld\n", $final_ovs_begin, $final_ovs_count
    disable 2
  end
  continue
end

# RGB extraction is complete here and the packed records contain the exact
# colors and normalized uint16 qualities consumed by Gamma + ABS.
break *0x55555571c7af
commands
  silent
  if $final_ovs_begin != 0
    set $final_ovs_end = $final_ovs_begin + $final_ovs_count * 40
    dump binary memory /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_same_run/vendor_final_ovs.bin $final_ovs_begin $final_ovs_end
    printf "DUMPED_COLORED_FINAL_OVS bytes=%ld\n", $final_ovs_count * 40
    disable 3
  end
  continue
end

continue
