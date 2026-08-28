# Z1 高分辨率 AdaptiveVoxelFilter 二进制证据

## 当前可执行结论

这次证据足以支持一项明确的主链修正，但不支持把任何猜测性的
`unordered_map` 迭代顺序写进主链。

应合入的行为是：

1. `HASH_MAP_FIRST_POINT` 使用命中点坐标的
   `floor(point / resolution)` 体素键；每个体素保留输入中的第一条完整
   24-byte range measurement。
2. 体素结果按原始输入顺序稳定输出，不按 hash bucket 顺序输出。
3. `index_filter_random=false` 且 `N > M` 时，精确索引是：

   ```text
   selected[k] = input[floor(k * N / M) + 1],  k = 0 .. M-1
   ```

   它会跳过 `input[0]`，不是当前实现中的
   `input[floor(k * N / M)]`。

在官方 node0 的真实 AdaptiveVoxelFilter 输入上，上述公式将结果从当前
均匀索引公式的集合交集 `2426/5000` 提升为完整记录、集合和顺序
`5000/5000`；最终命中点 XYZ 与冻结 node0 protobuf 也按 float32 位模式及
顺序 `5000/5000` 完全一致。这是明显提升，建议合入。

不要合入以下行为：

- 不要改成 `round`/`rint`；node0 上它只生成 7504 个体素点，且与官方
  7574 点集合的交集只有 3000。
- 不要按 hash 容器迭代顺序输出。完整 24-byte 记录反查证明官方输出对应
  的原输入索引严格单调，下降次数为 0。
- 不要把 59184 点直接送入本公式并声称已经对齐。官方动态探针显示，进入
  这个高分辨率 AdaptiveVoxelFilter 的实际输入是 44968 点；59184 点位于
  更早的原始/预过滤边界。若主链当前在 59184 点上直接做 `.4 m` 体素化，
  还必须先单独对齐它前面的 `filter_hybrid_grid_points`/范围输入边界。

## 保护范围

本任务没有修改以下内容：

- 主源码；
- `/opt/NavVis`；
- `/media/cybergeo/12T/CSSJ/datasets_rec/2026-02-08_07.33.20`；
- `/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/2026-02-08_07.33.20`。

新增内容只有本报告、只读 GDB 捕获脚本和离线验证脚本。

## 探针边界

目标二进制：

```text
/opt/NavVis/slam/lib/surveyor_ros/surveyorslam_processing_node
SHA256 8c7f7ea8180f9506468bb66fc3df9a3f27576c9a30ae056797f11637ab715ca6
```

设备配置是 Z1/G11 `slam_config_offline.lua`，输入是官方 RewriteStep 产生的
五个重写 bag。动态地址均为剥离 ELF 的相对虚拟地址：

| 边界 | RVA | 作用 |
|---|---:|---|
| Adaptive filter 总入口 | `0x62e770` | 读 options，串联 grid/index 两阶段 |
| HASH_MAP_FIRST_POINT adaptive helper | `0x638730` | 长度选择 |
| HASH_MAP_FIRST_POINT 单长度 kernel | `0x6381d0` | 体素首点集合 |
| deterministic index filter | `0x62d920` | 非随机定长抽样 |

捕获脚本：

```text
code/slam_probes/capture_adaptive_voxel.gdb
```

离线复现脚本：

```text
code/slam_probes/probe_adaptive_voxel.py
```

## options 的动态值

总入口的 options 对象动态读取结果：

| 字段 | 值 |
|---|---:|
| `max_length` | float32 `0.4000000059604645` |
| `min_length` | float32 `0.019999999552965164` |
| `max_range` | float32 `60.0` |
| `max_num_points` | float32 `5000.0` |
| `max_num_iterations` | `10` |
| `voxel_filter_type` | `2` = `HASH_MAP_FIRST_POINT` |
| `grid_filtering_enabled` | true |
| `index_filtering_enabled` | true |
| `index_filter_random` | false |

实际总入口点数是 `44968`，不是 `59184`。

24-byte 输入记录由两个 `Vector3f` 组成。动态数据中前 12 bytes 是逐射线原点，
后 12 bytes 是命中点；冻结 node cloud 保存的是后一个 `Vector3f`。

## (1) 体素键和数值精度

### floor/round

对 44968 条完整输入记录，仅用后 12 bytes 的命中点计算：

```text
key = floor(hit_xyz / float32(0.4))
```

然后按输入顺序保留每个 key 第一次出现的完整记录，结果为 7574 条，与二进制
进入 index filter 的 7574 条记录逐 byte、逐顺序完全一致。

对照数字：

| key 方案 | 体素数 | 与官方 7574 集合交集 | 顺序完全一致 |
|---|---:|---:|---:|
| float32 `floor` | 7574 | 7574 | 是 |
| promoted-double `floor` | 7574 | 7574 | 是 |
| float32 `round/rint` | 7504 | 3000 | 否 |
| trunc toward zero | 7245 | 7245 | 否 |

因此 `floor` 与 `round` 已无歧义。node0 不包含一个能让 float32-floor 和
promoted-double-floor 产生不同整数键的边界点，所以仅凭点集不能进一步区分
二者；但静态指令在 kernel 构造阶段明确把 float32 resolution 用
`cvtss2sd` 提升为 double，并保存 double resolution/reciprocal。最保守且已
逐点验证的表述是：输入坐标和配置值来自 float32，键语义是 floor；对 node0，
按 float32 运算或把这些 float32 值提升为 double 均产生完全相同的 7574 点。

官方首批保留的原输入索引为：

```text
0, 4, 25, 41, 44, 51, 53, 54, 56, 61, 68, 69, 77, 78, 88, 102
```

## (2) adaptive length 搜索

静态控制流和动态调用共同恢复出 Z1 本配置的流程：

```text
if input_count <= max_num_points or max_num_iterations == 0:
    return input

if max_num_iterations == 1:
    return voxel_filter((max_length + min_length) / 2)

sparse = voxel_filter(max_length)
dense  = voxel_filter(min_length)

if len(sparse) > max_num_points:
    return sparse

最多 10 次：
    若 (len(dense) - max_num_points) / max_num_points <= 0.1，停止
    middle = (max_bound + min_bound) / 2
    candidate = voxel_filter(middle)
    若 len(candidate) < max_num_points：
        max_bound, sparse = middle, candidate
    否则：
        min_bound, dense = middle, candidate
return dense
```

node0 的动态调用只有两次：

| 调用 | length | 输出数 |
|---:|---:|---:|
| 0 | `0.4000000059604645` | 7574 |
| 1 | `0.019999999552965164` | 44968 |

因为 `.4 m` 结果仍为 `7574 > 5000`，二进制直接采用 `.4 m` 结果，不执行任何
中点搜索，随后交给 index filter 截到 5000。也就是说，对 node0 来说，修改
二分搜索不会提升结果；真正的差异在 index filter 和正确的输入边界。

## (3) HASH_MAP 容器与输出顺序

`0x6381d0` 静态代码构造 8 个独立 hash 分片。内部 hash bucket/分片负责“该
体素是否已出现”的成员查询，但最终结果不是遍历 bucket 输出。二进制对原始
输入再做稳定保留，输出每个体素第一次出现的完整记录。

完整 24-byte 记录反查结果：

```text
grid output -> adaptive input: 7574/7574 exact
raw index descents:            0
monotonic segments:            1
```

此前观察到“多个单调段”是只按 XYZ/近邻反查造成的歧义；输入中存在重复或极近
的坐标，但完整记录反查没有歧义。因而无需、也不应复刻某个未证明的 hash
bucket 迭代顺序。

## (4) non-random index filter

对 node0：

```text
N = 7574
M = 5000
index[k] = floor(k * 7574 / 5000) + 1
```

前 16 个索引：

```text
1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23
```

最后 10 个索引：

```text
7559, 7561, 7562, 7564, 7565, 7567, 7568, 7570, 7571, 7573
```

与当前 `floor(k*N/M)` 的同输入对照：

| 指标 | 当前公式 | 恢复公式 |
|---|---:|---:|
| 输出数 | 5000 | 5000 |
| 与官方完整记录集合交集 | 2426 | 5000 |
| 相同顺序位置 | 0 | 5000 |
| 与冻结 XYZ float32 位模式顺序相同 | 否 | 5000/5000 |

用户此前报告的 `3196/5000` 是在 59184 点更早边界上的结果，不是本次同输入
控制实验。两组数字不矛盾；本次 `2426 -> 5000` 隔离了 index filter 本身。

## 冻结结果闭环

动态 AdaptiveVoxelFilter 输出的每条 24-byte 记录，其后一个 Vector3f 与冻结
文件：

```text
/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/
2026-02-08_07.33.20/internal/nodes/trajectory_node_clouds/
trajectory_node_clouds_00000000.zip
```

中的 `trajectory_node_00000000.pb` 逐 float32 位模式、逐顺序完全相同。

离线验证输出：

```json
{
  "adaptive_input_count": 44968,
  "grid_count": 7574,
  "output_count": 5000,
  "grid_records_exact": true,
  "output_records_exact": true,
  "frozen_xyz_float32_exact": true,
  "captured_output_sha256": "99f7fd288ae51341a5f9e3368bdecc639e715c5d334d5ba9efa364ac9efc22b0",
  "predicted_output_sha256": "99f7fd288ae51341a5f9e3368bdecc639e715c5d334d5ba9efa364ac9efc22b0"
}
```

复验命令：

```bash
python3 code/slam_probes/probe_adaptive_voxel.py \
  --capture-dir /tmp/navvis-adaptive-probe.dKvPf9/capture \
  --frozen-node-clouds \
  /media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/2026-02-08_07.33.20/internal/nodes/trajectory_node_clouds/trajectory_node_clouds_00000000.zip
```

动态 capture 位于 `/tmp`，重启后可能消失；报告中的算法、计数、哈希与验证
脚本保留在代码资源目录。
