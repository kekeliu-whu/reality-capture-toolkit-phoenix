set pagination off
set print thread-events off
set confirm off
set disable-randomization on
starti
set scheduler-locking on

# The first exposure OptimalViewSelection worker owns a score-cloud adapter at
# selector+0x90.  Its inner PCL cloud stores 48-byte PointXYZRGBINormal records.
break *0x555555725f80
continue
set $selector = $rdi
set $cloud = *(void**)($selector + 0x90)
set $inner = *(void**)($cloud + 8)
set $begin = *(unsigned char**)($inner + 0x30)
set $end = *(unsigned char**)($inner + 0x38)
printf "EXPOSURE_CLOUD selector=%p cloud=%p inner=%p begin=%p end=%p stride=48 count=%ld\n", $selector, $cloud, $inner, $begin, $end, ($end-$begin)/48
dump binary memory /media/cybergeo/12T/CSSJ/resources/navvis-cleanroom-alignment/work/color_alignment/official_single_exact_20260827/exposure_cloud_48b.bin $begin $end
quit
