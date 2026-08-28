set pagination off
set confirm off
set disable-randomization on
set environment OMP_NUM_THREADS 1

# Pinned to nv_sparse-depthmap-renderer build-id
# 24ef8bee2e35486ec1e9922dd5459cd028bbfd20.
starti
set $image_base = 0x555555554000
set $query_count = 0

break *($image_base + 0x138d20)
commands
  silent
  set $query_count = $query_count + 1
  if $query_count == 1
    set $raytracer = $rdi
    set $octree = *(void**)($raytracer + 0x28)
    set $cloud = *(void**)($raytracer + 0x8)
    set $point_begin = *(void**)($cloud + 0x30)
    set $point_end = *(void**)($cloud + 0x38)
    printf "FIRST_QUERY raytracer=%p octree=%p\n", $raytracer, $octree
    printf "CLOUD object=%p points=%p count=%ld is_dense=%u\n", $cloud, $point_begin, ($point_end-$point_begin)/16, *(unsigned char*)($cloud+0x50)
    printf "CLOUD_POINT index=0 xyz=(%.9g %.9g %.9g)\n", *(float*)$point_begin, *(float*)($point_begin+4), *(float*)($point_begin+8)
    printf "CLOUD_POINT index=2492855 xyz=(%.9g %.9g %.9g)\n", *(float*)($point_begin+2492855*16), *(float*)($point_begin+2492855*16+4), *(float*)($point_begin+2492855*16+8)
    printf "CLOUD_POINT index=2514244 xyz=(%.9g %.9g %.9g)\n", *(float*)($point_begin+2514244*16), *(float*)($point_begin+2514244*16+4), *(float*)($point_begin+2514244*16+8)
    printf "OCTREE depth=%u resolution=%.17g bounds=[%.17g %.17g %.17g]..[%.17g %.17g %.17g]\n", *(unsigned int*)($octree+0x24), *(double*)($octree+0x60), *(double*)($octree+0x68), *(double*)($octree+0x78), *(double*)($octree+0x88), *(double*)($octree+0x70), *(double*)($octree+0x80), *(double*)($octree+0x90)
    printf "OCTREE_PREFIX\n"
    x/40gx $octree
  end
  continue
end

# Immediately after getIntersectedVoxelIndices() has filled the local
# std::vector<int> at rsp+0x10.
break *($image_base + 0x138d91)
commands
  silent
  if $query_count == 1
    set $candidate_begin = *(int**)($rsp + 0x10)
    set $candidate_end = *(int**)($rsp + 0x18)
    set $candidate_count = ($candidate_end - $candidate_begin)
    printf "FIRST_CANDIDATES begin=%p end=%p count=%ld\n", $candidate_begin, $candidate_end, $candidate_count
    x/32dw $candidate_begin
    dump binary memory work/panorama_alignment_20260827/sparse_renderer_gdb_candidates/first_ray_candidates.i32 $candidate_begin $candidate_end
    disable 1
    disable 2
  end
  continue
end

# Raw SIMD min/max immediately before the PCL bounding-box expansion.
break *($image_base + 0x13c850)
commands
  silent
  printf "RAW_POINT_BOUNDS min=(%.9g %.9g %.9g) max=(%.9g %.9g %.9g)\n", *(float*)$rsp, *(float*)($rsp+4), *(float*)($rsp+8), *(float*)($rsp+0x10), *(float*)($rsp+0x14), *(float*)($rsp+0x18)
  disable 3
  continue
end

continue
