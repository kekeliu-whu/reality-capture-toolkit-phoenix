set pagination off
set print thread-events off
set confirm off
set disable-randomization on
set logging file /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_black_filter_20260827/official_single_thread/vendor_capture.log
set logging overwrite on
set logging redirect on
set logging enabled on
set $final_ovs_begin = 0
set $final_ovs_count = 2857623
starti

# The SIMD quality-count loop is reached after the final OVS worker and keeps
# the 40-byte SelectedViews_<5> base in r12.  Restrict by the full cloud count
# because the exposure worker reaches the same instruction first.
break *0x555555725a83
commands
  silent
  if $rbp == $final_ovs_count
    set $final_ovs_begin = (unsigned char*)$r12
    printf "FINAL_OVS_SELECTED begin=%p count=%ld\n", $final_ovs_begin, $final_ovs_count
    disable 1
  end
  continue
end

# RGB extraction and normalized quality encoding are complete here.
break *0x55555571c7af
commands
  silent
  if $final_ovs_begin != 0
    set $final_ovs_end = $final_ovs_begin + $final_ovs_count * 40
    dump binary memory /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/gamma_black_filter_20260827/official_single_thread/vendor_final_ovs.bin $final_ovs_begin $final_ovs_end
    printf "DUMPED_FINAL_OVS bytes=%ld\n", $final_ovs_count * 40
    disable 2
  end
  continue
end

continue
