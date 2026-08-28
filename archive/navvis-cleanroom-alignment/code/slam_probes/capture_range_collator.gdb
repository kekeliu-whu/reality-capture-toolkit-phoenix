set pagination off
set confirm off
starti
set $image_base = 0x555555554000
set $hits = 0
break *($image_base + 0x6426e0)
commands
  silent
  set $begin = *(void **)($rdx + 8)
  set $end = *(void **)($rdx + 16)
  set $count = ((char *)$end - (char *)$begin) / 28
  printf "COLLATOR hit=%d time_ticks=%lld rays=%lld sensor_arg=%p\n", $hits, *(long long *)$rdx, $count, $rsi
  if $count > 0
    printf "  first xyzit=%g,%g,%g,%g,%g\n", *(float *)$begin, *(float *)($begin+4), *(float *)($begin+8), *(float *)($begin+12), *(float *)($begin+24)
    set $last = (char *)$end - 28
    printf "  last  xyzit=%g,%g,%g,%g,%g\n", *(float *)$last, *(float *)($last+4), *(float *)($last+8), *(float *)($last+12), *(float *)($last+24)
  end
  set $hits = $hits + 1
  if $hits >= 24
    quit
  end
  continue
end
continue
