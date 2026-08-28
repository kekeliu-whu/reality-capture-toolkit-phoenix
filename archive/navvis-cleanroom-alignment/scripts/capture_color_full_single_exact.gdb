set pagination off
set print thread-events off
set confirm off
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/official_single_exact_20260827/vendor_capture.log
set logging overwrite on
set logging redirect on
set logging enabled on
set $final_ovs_begin = 0
set $final_ovs_count = 2857623
starti

# Dump the solved GammaModel map used by this process.  The helper prints the
# model values, while the explicit map walk preserves the view identifier.
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

# Capture the final SelectedViews_<5> allocation after color extraction and
# normalized-quality encoding have completed.
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

break *0x55555571c7af
commands
  silent
  if $final_ovs_begin != 0
    set $final_ovs_end = $final_ovs_begin + $final_ovs_count * 40
    dump binary memory /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/official_single_exact_20260827/vendor_final_ovs.bin $final_ovs_begin $final_ovs_end
    printf "DUMPED_FINAL_OVS bytes=%ld\n", $final_ovs_count * 40
    disable 3
  end
  continue
end

# Preserve the exact partition consumed by KNN fallback in this same process.
break *0x5555557a5977
commands
  silent
  set $colored_begin = *(char**)$r13
  set $colored_end = *(char**)($r13 + 8)
  set $uncolored_begin = *(char**)$r12
  set $uncolored_end = *(char**)($r12 + 8)
  printf "KNN_PARTITION colored=%ld uncolored=%ld\n", ($colored_end-$colored_begin)/4, ($uncolored_end-$uncolored_begin)/4
  dump binary memory /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/official_single_exact_20260827/vendor_knn_colored_indices.i32 $colored_begin $colored_end
  dump binary memory /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/official_single_exact_20260827/vendor_knn_uncolored_indices.i32 $uncolored_begin $uncolored_end
  disable 4
  continue
end

continue
