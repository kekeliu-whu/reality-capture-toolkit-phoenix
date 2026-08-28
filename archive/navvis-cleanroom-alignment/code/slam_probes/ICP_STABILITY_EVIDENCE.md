# ConstraintBuilder `ICP_STABILITY(5)` 二进制证据

## 结论

分析对象：`/opt/NavVis/slam/lib/surveyor_ros/compute_constraints`，SHA-256
`31bb94da979d9b4ef3587fb21870ee1c5119b249e99948bd1f89e64b6d702fb9`，Build ID
`cc5a5581c38516e88571b08ad5cf462c1417eb89`。

稳定性判定的直接输入不是 ICP 的 `converged` 标志，而是最终 ICP 点面残差在二进制局部
6-DoF 参数化下得到的 `6×6 double` 对称正规/信息矩阵 `H`。令最终点面残差的局部
Jacobian 为 `J`；在本次配置中预测先验权重为 0、Huber 关闭、对应权重均为 1，故输入量为：

```text
H = Jᵀ J = Σᵢ Jᵢᵀ Jᵢ
eigvalsh(H) = [λ0, λ1, λ2, λ3, λ4, λ5],  λ0 <= ... <= λ5

constraint_strength   = λ0
constraint_anisotropy = λ5 / λ0
```

这里的 `Jᵢ` 是二进制对最终 ICP 位姿和最终三层 surfel correspondence 重新求得的点面残差
Jacobian；它的精确坐标规约见下一节，不能用未归一化坐标中的 `[p×n, n]` 直接替换。
`(submap2,node1120)` 的三层
correspondence 数是 `1976 + 397 + 46 = 2419`，序列化权重全为 1；`H` 的平移块迹为
`2419.000080...`，确认没有按 correspondence 数取平均。

标准参数为：

```text
max_constraint_anisotropy               Amax  = 50.0
min_constraint_strength                  Smin  = 50.0
max_constraint_anisotropy_tracking_good Agood = 0.0
```

精确判定是：

```text
is_stable = eigen_decomposition_ok
         && (constraint_anisotropy <  Amax)   # 严格小于
         && (constraint_strength   >= Smin)   # 包含等号
```

因此各向异性恰好等于 50 时拒绝，强度恰好等于 50 时接受。`IcpResult.converged` 不参与这个
布尔式。

`tracking_quality` 的精确分支为：

```text
if !is_stable:                          q = 0
else if constraint_anisotropy <= Agood: q = 1
else: q = float(1 - (constraint_anisotropy - Agood)
                    / (Amax - Agood))
```

标准参数下，稳定约束即 `q = float(1 - constraint_anisotropy / 50)`；结果最后收窄为
protobuf `float`。

## 二进制位置

- `0x1f3c9f...`：最终 ICP 迭代把 36 个 double 写入信息矩阵。
- `0x225599`：把该矩阵、三个 stability 参数及结果对象传给稳定性例程。
- `0x223000..0x223677`：稳定性例程；`0x223491` 调用 6×6 自伴随特征分解。
- `0x22357a; jbe 0x223670`：若 `Amax <= anisotropy` 则拒绝，故通过条件严格为
  `anisotropy < Amax`。
- `0x22358c; jb 0x223670`：若 `strength < Smin` 则拒绝，故通过条件为
  `strength >= Smin`。
- `0x2235ab..0x2235db`：上述 tracking-quality 分段线性公式；`0x898344` 是 `1.0f`。
- `0x2235ed..0x22362a`：写出 `is_stable`、quality、特征值和两个指标。

## `normalize_icp=true` 的 Jacobian 与尺度

### 已有充分证据、可直接编码的部分

`0x1f2da0` 的正规方程 kernel 接收一个固定刚体变换
`T_N=(R_N,t_N)`。以下公式对传入的 `T_N` 是确定的，并已用 GDB 对 2419 条最终
correspondence 逐条复算；复算矩阵与二进制栈上 36 个 double 的最大绝对差为 **0**。

对第 `i` 条 correspondence，`p_i,n_i` 是其第一个索引选中的 surfel 点和法向。先在
`float` 中计算：

```text
u_i = float32(R_N p_i + t_N)
v_i = float32(R_N n_i)                 # 不乘尺度，也不在本 kernel 内重新单位化

mu  = (1/N) * sum_i double(sqrtf(u_ix*u_ix + u_iy*u_iy + u_iz*u_iz))
s_d = 1.0 / mu                         # double
s   = float32(s_d)                     # 主循环实际使用的尺度
p'_i = float32(s * u_i)
n'_i = v_i
```

平方和在 SSE 中的实际结合顺序是 `(u_z*u_z + u_y*u_y) + u_x*u_x`；每个乘加和
`sqrt` 均为 `float`，每个半径再转成 `double` 顺序累加。若 `normalize_icp=false`，则
`s_d=s=1`；若 `1/mu` 非有限或超过 `DBL_MAX`，也回退到 1。

局部增量顺序和单行 Jacobian 为：

```text
delta = [dtheta_x, dtheta_y, dtheta_z, dt_x, dt_y, dt_z]

c_i = p'_i cross n'_i
J_i = [c_ix, c_iy, c_iz, n'_ix, n'_iy, n'_iz]
H   = sum_i w_i * J_i^T J_i            # 本样本全部 w_i=1，不除以 N
```

叉乘是在 `p'`、`n'` 的六个 `float` 分量各自提升为 `double` 后计算，`H` 也按
correspondence 顺序以 `double` 累加。二进制使用的是 **`[p'×n', n']`**，不是此前尝试的
`[n', p'×n']`；后者虽不改变特征值，却会改变矩阵的参数块排列。只给 source 点旋转、
但不施加这里的固定平移和均值半径尺度，也不会得到同一信息矩阵。

对应反汇编证据：

- `0x1f2ec9 -> 0x1f3ea0`：进入 `normalize_icp` 分支。
- `0x1f3f18..0x1f3ff8`：只对每条 correspondence 的第一个索引点施加 `T_N`，计算并
  以 `double` 累加半径。
- `0x1f4023..0x1f4054`：除以 `N`，再计算 `1/mu`。
- `0x1f323a..0x1f3351`：点坐标乘 `float32(s)`；法向只旋转、不乘 `s`。
- `0x1f33d2..0x1f3454`：按 double 计算 `p'×n'`。
- `0x1f3491..0x1f3905`：按 `[p'×n',n']` 外积累加 36 项。
- `0x1f3c83..0x1f3dxx`：把最终 36 个 double 写到输出信息矩阵。

### `(submap2,node1120)` 数值验证

六次收缩迭代中，固定 `T_N` 不变，最终 correspondence 数和尺度依次为：

```text
N = 2782, 2741, 2708, 2659, 2592, 2419
s = 0.1747330874, 0.1743683368, 0.1740491688,
    0.1738995463, 0.1729860455, 0.1718030274
```

最终一次：

```text
mu  = 5.8206192650741757
s_d = 0.17180302549599186
s   = 0.17180302739143372             # float32

t_N = [-0.873173356, -13.3956337, -0.515120149]       # float32
q_N = [0, 0, 0.202297956, 0.979324043] (x,y,z,w)       # float32
```

第一条最终 correspondence 的运行时量为：

```text
p' = [-0.135115862, 0.00947608892, -0.317937464]
n' = [-0.0227875132, 0.00734042423, 0.99971348]
J  = [0.0118071697, 0.142322153, -0.000775871246,
      -0.0227875132, 0.00734042423, 0.99971348]
```

独立重跑得到的逐条复算结果为：

```text
GDB correspondence count = 2419
max_abs(recomputed_H - binary_H) = 0

eigvalsh(H) = [
  35.301354944450225,
  385.85748185152914,
  450.97599241066797,
  751.5513678605735,
  874.7984104005076,
  1582.3963738339367
]
```

它与前一次官方捕获
`[35.301353757,385.857479862,450.975998973,751.551370365,874.798395140,1582.396378102]`
的最大绝对差为 `1.5260492e-05`；而同一次运行内，按上述公式复算 36 项的差为严格 0。
因此此前出现 `127481/351551` 或 `32554` 的大特征值不是官方矩阵的微小数值漂移，而是
缺少 `T_N` 平移及均值半径尺度所致。

### 尚不建议硬编码的部分

本样本中，`T_N` 与 FCS 输出 pose `(t,q)` 满足：

```text
d   = sqrt(q_z^2 + q_w^2)
q_N = (0, 0, -q_z/d, q_w/d)
t_N = -R(q_N) t
```

代入 diagnostic protobuf 的 FCS pose 后，转成 `float32` 与上面的 GDB 值逐位一致。这是
“取 z/w、归一化为纯 yaw 后求逆”的强数值证据；但本任务没有继续向上追完调用方对任意
roll/pitch pose 构造 `T_N` 的全部分支，也没有第二组独立姿态样本。因此可以合入
“给定 `T_N` 后的 normalize/Hessian kernel”，**暂不要把上述 FCS-pose 到 `T_N` 的推导
作为通用规则硬编码进主流程**。本任务按要求没有修改主源码。

## `(submap2,node1120)` 验证

证据文件：`/tmp/navvis-constraint-score.yFAZv6/out/constraint_data/constraint_data_00000000.zip`。
GDB 在稳定性例程返回时读到内部 double：

```text
λ = [
  35.301353757045440,
  385.85747986209202,
  450.97599897262751,
  751.55137036454437,
  874.79839514001549,
  1582.3963781017824
]

strength   = 35.301353757045440
anisotropy = 1582.3963781017824 / 35.301353757045440
           = 44.825373808390218
```

protobuf 收窄后的值为 `strength=35.301353454589844`、
`anisotropy=44.825374603271484`。标准门限下各向异性通过 (`44.825... < 50`)，强度失败
(`35.301... < 50`)，所以 `is_stable=false`、`tracking_quality=0`、最终
`failure_reason=ICP_STABILITY(5)`。该 ICP 为 `converged=false, num_iterations=6`；这不是
拒绝原因。

只改变诊断配置、保持候选和数据不变的边界夹逼结果：

| 独立门限实验 | 结果 |
|---|---|
| `Smin=0, Amax=44` | `ICP_STABILITY(5)` |
| `Smin=0, Amax=45` | 通过，`q=0.00388058205` |
| `Smin=35, Amax=1000000` | 通过 |
| `Smin=36, Amax=1000000` | `ICP_STABILITY(5)` |

这四次运行仅写入 `/tmp`，未改 `/opt/NavVis`、数据集或冻结参考；严格/非严格等号语义由
上面的 `jbe`/`jb` 指令确定。

## 冻结 accepted constraints 交叉验证

冻结 archive 共 15 条，全部是 `valid=true`、`failure_reason=NONE(6)`、
`is_stable=true`，同时全部记录为 `converged=false, num_iterations=6`：

```text
constraint_anisotropy range = [8.27132797241211, 38.508750915527344]  (< 50)
constraint_strength range   = [64.5873794555664, 227.13092041015625]  (>= 50)
```

15 条记录的 protobuf `tracking_quality` 与 `1 - anisotropy / 50` 的最大绝对差为
`4.768371586472142e-08`（float 舍入量级）。这同时排除了把 ICP convergence、
correspondence 数或平均 residual 直接当作 `ICP_STABILITY` 门限量的解释。
