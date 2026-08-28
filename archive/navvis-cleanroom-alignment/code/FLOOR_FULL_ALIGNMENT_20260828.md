# Floor 完全对齐验收（2026-08-28）

## 结论

当前 `src/navvis_recon/floor_estimator.py` 在可用的完整同输入语料上为 `EXACT`：扫描
`/media/cybergeo/12T/DT` 得到 220 组原版 `artifacts/trace.csv` 与
`artifacts/floors.json`，共 1,796,815 条 trace、615 个参考 floor；重构结果 220/220 组
JSON 对象逐字段一致，floor 总数同为 615，mismatch 为 0。

这里的 `EXACT` 指最终顶层数组、每个 floor 的 `z_min`、`z_max`、每一段纳秒
`time_ranges` 及其顺序都相同，不只是 floor 数量或近似高度相同。机器报告位于
`../work/current_code_reacceptance_20260828/floor_full_alignment_score.json`。

## 已恢复的行为

- SimpleFloorEstimator 是按 trace 时间顺序维护 current/previous floor 的状态机，不是全局
  z 聚类；支持离开后再次回到已有楼层并生成多段 time range。
- 高度范围使用 0.1 m bin、3.0 m 标准层高、4.0 m 最大跨度、2.1 m 双层判定、0.03 m
  容差；负高度 bin 使用向零截断。
- refiner 顺序为 tiny merge、adjacent merge、double-floor split、再次 tiny merge；time range
  小于 195 ms 的间隔会合并，tiny 判据包含 5 s 总时长和 0.06 m 高度范围。
- adjacent merge 只验证边界两侧指定 histogram bin 的较大 cluster，并要求两层访问时间不
  重叠；不能把边界附近任意显著 bin 当作通过。
- double-floor split 使用 1.5 cm 重叠 hard limits，然后以所有现有 floor 的冻结边界为种子，
  对完整 trace 重新执行状态机。只重放被拆 floor 自己的点会遗漏跨边界归属变化。
- 输出采用原版 schema：`artifacts/floors.json` 是顶层数组，不含 `index`、`z_center` 或
  `floors` wrapper。

## runner 接线验收

`runner/navvis_postprocessing_recon.py` 已移除旧 `floor_summary` 路径。runner 现在从原始
非 laser bag 的 `/imu/magnetic_field` 取得纳秒时钟，在优化轨迹支持区间内按原版周期抽样，
插值 XYZ/姿态，以 6 个有效数字序列化 `artifacts/trace.csv`，再运行 refined estimator 并
写出 `artifacts/floors.json`。

两个 G11-0109 真实数据的 runner 输入链验收如下：

| 数据 | trace | floors | 结果 |
|---|---:|---:|---|
| `2026-07-21_11.07.05` | 5,374 行 | 1 | `trace.csv` byte-exact；`floors.json` exact |
| `2026-01-19_19.04.51` | 17,878 行 | 5 | `trace.csv` byte-exact；`floors.json` exact |

另以 G10-512 `2023-05-15_10.18.42` 做兼容 smoke：从原始 magnetic-field bag 和
`trajectory_slam.bag` 生成 670 条 trace、1 个 floor，流程正常结束；该数据因原版许可过期
没有可用于 exact 比较的官方 Floor 结果，因此不把 G10 smoke 计入 exact 分母。

缺少 magnetic-field bag、无重叠时间、非有限轨迹或非严格递增时间戳时，runner 会直接报错，
不再回退到旧高度聚类或静默生成结构错误的 `floors.json`。

## 守门命令

```bash
PYTHONPATH=code/src code/tools/evaluate_floor_alignment.py \
  /media/cybergeo/12T/DT \
  --output work/current_code_reacceptance_20260828/floor_full_alignment_score.json

PYTHONPATH=code/src python3 -m unittest discover -v \
  -s code/tests -p 'test_floor_estimator.py'
```

首条命令应输出 `classification=EXACT`、`exact_case_count=220`、
`mismatch_case_count=0`；第二条命令当前为 4/4 通过。
