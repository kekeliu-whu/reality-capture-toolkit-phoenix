set pagination off
set confirm off
set print thread-events off
set breakpoint pending on
set disable-randomization on

starti

break *0x5555557256f0
commands
  silent
  printf "OVS_SELECT_ENTRY this=%p views=%p output=%p\n", $rdi, $rsi, $rdx
  printf "OVS_THIS\n"
  x/24gx $rdi
  set $adapter = *(void**)($rdi + 0x90)
  printf "OVS_ADAPTER object=%p vtable=%p\n", $adapter, *(void**)$adapter
  x/24gx $adapter
  set $cloud = *(void**)($adapter + 0x08)
  printf "OVS_ADAPTER_CLOUD object=%p begin=%p end=%p\n", $cloud, *(void**)($cloud + 0x30), *(void**)($cloud + 0x38)
  x/16gx $cloud
  printf "OVS_ADAPTER_VTABLE\n"
  x/12gx *(void**)$adapter
  quit
end

continue
