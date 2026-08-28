set pagination off
set confirm off
starti
break *0x5555557c6770
continue
set $residuals = $rdx
printf "EXACT_ENTRY this=%p residuals=%p num_residuals=%d\n", $rdi, $residuals, *(int*)($rdi+32)
disable 1
finish
printf "EXACT_RESIDUALS\n"
x/9gf $residuals
quit
