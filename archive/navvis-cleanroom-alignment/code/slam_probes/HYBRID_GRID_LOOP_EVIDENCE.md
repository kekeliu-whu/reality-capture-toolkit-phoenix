# HybridGrid / 3D loop-constraint evidence

调查对象是 G11 本机二进制
`/opt/NavVis/slam/lib/surveyor_ros/compute_constraints`，SHA-256：
`31bb94da979d9b4ef3587fb21870ee1c5119b249e99948bd1f89e64b6d702fb9`，
Build ID：`cc5a5581c38516e88571b08ad5cf462c1417eb89`。数据证据来自冻结的
`datasets_proc_reference_g11_0109/2026-02-08_07.33.20`。本文只陈述可以由二进制、
protobuf 或冻结输出复核的结论。

## 1. HybridGrid 的值、坐标和概率

NavVis 二进制嵌入了 Cartographer 的源路径、protobuf descriptor 和概率查表；其常量与
Cartographer 官方 `probability_values.h/.cc`、`mapping/3d/hybrid_grid.h` 一致：

```text
kMinProbability          = 0.1f
kMaxProbability          = 0.9f
kUnknownProbabilityValue = 0
kUpdateMarker            = 32768
kValueCount              = 32768
scale                    = 0.8f / 32766.f
```

精确映射为：

```text
P(0) = 0.1f
P(v) = 0.1f + (v - 1) * 0.8f / 32766.f,  1 <= v <= 32767
P(v + 32768) = P(v)                         （更新标记期间）

V(p) = round((clamp(p, 0.1, 0.9) - 0.1) * 32766 / 0.8) + 1
```

点到体素索引不是 `floor`，而是逐轴 `RoundToInt(point / resolution)`；索引对应的体素中心
是 `index * resolution`。官方源码：

- https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/probability_values.h
- https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/probability_values.cc
- https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/3d/hybrid_grid.h

冻结 `HybridGrid` descriptor 的字段为：`resolution=1`（float）、`x_indices=3`、
`y_indices=4`、`z_indices=5`（三者均为 packed sint32）、`values=6`（packed int32）。
2026-02-08 的 submap 0 高分辨率栅格是 `0.20000000298023224 m`、248795 个已存储
体素；低分辨率字段只保存 `0.44999998807907104 m` 分辨率，没有体素。

## 2. FastCorrelativeScanMatcher3D 分数

### 2.1 高分辨率分数

预计算栅格先把已知概率量化为：

```text
q(v) = RoundToInt((P(v) - 0.1) * 255 / 0.8), q in [0, 255]
```

未知预计算体素保持 0。每个候选位姿的分数是：

```text
high_score = 0.1 + (sum_i q(cell(T * point_i)) / N) * (0.8 / 255)
```

分支定界较粗层使用对应子块的最大 `q`；到叶子时使用上述候选分数。NavVis 标准参数是
`branch_and_bound_depth=8`、`full_resolution_depth=3`、`min_score=0.55`。高分门限是
严格的 `high_score > min_score`（实现以 `score <= min_score` 拒绝）。官方源码：

- https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/3d/scan_matching/precomputation_grid_3d.cc
- https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/3d/scan_matching/fast_correlative_scan_matcher_3d.cc

### 2.2 低分辨率分数

存在低分辨率点云和低分辨率 HybridGrid 时，定义是直接平均概率，没有插值或 uint8
量化：

```text
low_score(T) = (1 / M) * sum_j P(low_grid[RoundToInt((T * low_point_j) / r_low)])
```

未知体素按 `P(0)=0.1` 参与平均；门限是 `low_score >= min_low_resolution_score`。
官方实现：
https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/3d/scan_matching/low_resolution_matcher.cc

本批 G11 冻结数据走的是禁用分支：每个持久化节点有 5000 个高分点、低分点云字段为空，
submap 低分 grid 也没有体素。因此 NavVis 输出
`low_resolution_score=+inf`、`low_resolution_score_ok=true`，表示低分门限不适用，不能把
`rotational_score` 误认成低分辨率分数。

冻结 `FastCorrelativeScanMatcherResult` descriptor 的关键字段为：`score=1`、
`pose_estimate=2`、`rotational_score=3`、`low_resolution_score=4`、`score_ok=8`、
`low_resolution_score_ok=9`、`rotational_score_ok=10`、`matched=11`。

## 3. `ray_casting_contradiction_ratio` 的确切定义

计算函数位于该 PIE 的静态地址 `0x232800..0x232e17`；调用与结果写入位于
`0x2255cb..0x225603`；最终门限检查位于 `0x242174..0x2421b0`。二进制逻辑可还原为：

```text
bad = 0
free = 0
for each ray (origin, endpoint):
    a = RoundToInt(T(origin)   / grid_resolution)
    b = RoundToInt(T(endpoint) / grid_resolution)
    d = b - a
    n = max(abs(d.x), abs(d.y), abs(d.z))
    if n == 0: continue

    # k=0..n-1：包含原点体素，明确排除 endpoint 体素。
    for k in [0, n):
        # x86 signed idiv，负数向 0 截断；三轴分别计算。
        c = a + trunc_toward_zero(k * d / n)
        if c is outside HybridGrid storage or value(c) == 0:
            bad += 1
        elif P(value(c)) > occupied_probability_threshold:
            bad += 1
        else:
            free += 1

ratio = 0.0f if bad + free == 0 else float(bad) / float(bad + free)
```

所以它不是“发生过遮挡的射线条数 / 射线条数”，也不是端点占用率；分母是所有射线沿途
访问的体素样本数。未知、未分配块和越界体素都计入矛盾；已知且概率不高于阈值的体素
才计为自由。占用比较是严格 `>`。标准参数为
`ray_casting_occupied_probability_threshold=0.55`、
`max_ray_casting_contradiction_ratio=0.25`；最终仅在 ratio 字段存在且最大门限大于 0 时，
以严格 `ratio > 0.25` 拒绝。

射线检查用的是 `SubmapScanMatcher` 在运行时由 submap 支持节点构造的 HybridGrid，不是
`submap_clouds.field 1` 中仅持久化的表面栅格。日志对 submap 2 明确记录支持节点
`329 -> 33`（`searchable_node_subsampling_factor=10`）。直接拿冻结表面栅格代替该运行时
射线栅格会把大量自由空间错判为 unknown，数值不会等于官方 ratio。

相关 protobuf 字段：

- `ConstraintBuilderOptions.max_ray_casting_contradiction_ratio = 126`（double）
- `ConstraintBuilderOptions.ray_casting_occupied_probability_threshold = 127`（double）
- `ConstraintBuilderResult.valid_match = 10`（bool）
- `ConstraintBuilderResult.failure_reason = 11`（enum）
- `ConstraintBuilderResult.ray_casting_contradiction_ratio = 13`（float）
- mutable metadata 中 ratio 为字段 5
- `FailureReason.RAY_CASTING_CONTRADICTION = 10`

## 4. 2026-02-08 数值验证

冻结 accepted pair 选 `(submap=0, node=620)`：

```text
valid_match = true
failure_reason = NONE (6)
ray_casting_contradiction_ratio = 0.21503841876983643
0.25 - ratio = 0.03496158123016357
```

冻结 accepted archive 共 15 条，ratio 范围是
`[0.11612319201231003, 0.21503841876983643]`，全部低于 0.25。

对 `(submap=2, node=1120)` 的标准门限诊断：

```text
min_score = 0.55
valid_match = false
failure_reason = HIGH_RES_THRESH_GRID (1)
FCS matched = false, score_ok = false
low_resolution_score = +inf, low_resolution_score_ok = true
```

把诊断配置中的 `min_score` 单独降到 0.0 后，可以观测被标准高分门限挡住的候选：

```text
high score            = 0.5435789227485657  (< 0.55)
rotational score      = 0.9042913317680359
low-resolution score  = +inf
score_ok / low_ok / rotational_ok / matched = true / true / true / true
ray contradiction     = 0.1882352977991104  (< 0.25)
final failure          = ICP_STABILITY (5)
```

因此标准流程对 `(2,1120)` 的首个决定性差异是高分辨率 FCS 分数少
`0.006421077251434326`，不是低分辨率门限，也不是射线矛盾门限。强制越过高分门限后，
射线检查仍通过，后续由 ICP stability 拒绝。

## 5. 复核命令

只读探针不会生成或修改数据：

```bash
python3 code/slam_probes/probe_hybrid_loop.py

python3 code/slam_probes/probe_hybrid_loop.py \
  --diagnostic-zip /tmp/navvis-constraint-score.yFAZv6/out/constraint_data/constraint_data_00000000.zip
```

第二条命令中的 `/tmp` 产物是本次独立 `compute_constraints` 诊断结果；若临时目录已清理，
重新提供同类 diagnostic zip 即可。探针刻意不以冻结表面 grid 伪装运行时 ray grid。

## 6. Z1 全量 `SAVE_ALL` 重放

随后使用正确的 Z1 配置完整执行了
`RewriteStep -> CopyingPart1 -> SurveyorSLAM`。五个 rewrite 后的 bag 必须作为同一个
`--bags` 多值参数传入，两个 Hesai topic 同理：

```text
--bags control.bag vert_1.bag horiz_0.bag vert_0.bag horiz_1.bag \
--hesai-packets-topics /laser_horiz/packets /laser_vert/packets
```

原始加密雷达 bag 不能绕过 RewriteStep；CopyingPart1 还会提供 SLAM 所需的
`artifacts/origin.json`。标准全量重放得到 41 个 sampled candidates、15 个 valid；失败分布为
`NONE=15`、`THRESH_ROT=16`、`HIGH_RES_THRESH_GRID=6`、`ICP_STABILITY=4`。新生成的
1617 个 node、6 个 submap、点云归档、`optimization_state.pb` 和
`optimization_data.pb` 均与冻结参考逐归档/逐文件一致。

FCS 的真实开关是：

```text
skip_fcs = max(search_region.scaling) < 0.1 m
```

41 个候选中 17 个跳过、24 个执行，全部满足该规则。17 个跳过项为：

```text
(0,620) (0,720) (0,820)
(1,90) (1,190) (1,290) (1,390)
(0,870) (0,920)
(2,60) (2,160) (2,260) (2,360) (2,460) (2,560)
(3,1180) (3,1280)
```

其中 13 个有效；`(0,820)`、`(0,870)`、`(0,920)`、`(3,1280)` 在后续
ICP stability 阶段拒绝。24 个执行 FCS 的候选中，16 个先被旋转门限拒绝，6 个被高分辨率
栅格门限拒绝，只有 `(4,930)` 和 `(4,1610)` 有效；两者分数分别为
`0.7771055102348328` 和 `0.6547331213951111`。

只把 `min_score` 从 0.55 改为 0 的第二次全量重放恢复出六个标准低分项：

```text
(2,1120) 0.5444592833518982
(3,770)  0.2076134830713272
(4,830)  0.32363104820251465
(3,1460) 0.170960932970047
(3,1510) 0.1729612499475479
(3,1560) 0.1708812415599823
```

这六项继续进入 ICP 后全部由 stability 拒绝，没有新增有效约束，最终优化状态仍与标准
byte-exact。完整运行时临时产物位于本轮 `/tmp/navvis-z1-saveall2.ZzZUiH`；本文件保留其
不依赖临时路径的结论和复现参数。
