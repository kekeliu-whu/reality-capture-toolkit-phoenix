"""
IMU 时间同步和旋转对齐分析工具

该模块用于：
1. 读取两个IMU传感器的数据
2. 通过陀螺仪幅度的B-spline插值进行时间同步
3. 计算两个IMU之间的相对旋转矩阵
4. 生成详细的可视化分析结果
"""

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from scipy import signal
import matplotlib
from proto.sensors_pb2 import ImuMsgList

# ============================================================================
# 配置参数（需要在matplotlib后端设置前定义）
# ============================================================================

# 是否显示图形
SHOW_PLOTS = True

# 根据SHOW_PLOTS动态设置matplotlib后端
# 如果不显示图形，使用Agg后端（非GUI，速度快）
# 如果显示图形，使用默认交互式后端
if not SHOW_PLOTS:
    matplotlib.use("Agg")  # 使用非GUI后端，禁用窗口弹出
# else: 使用默认的交互式后端

import matplotlib.pyplot as plt

# 配置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]

# ============================================================================
# 其他配置参数
# ============================================================================

# 文件路径
IMU1_FILE = R"D:\slam\imu.dat"
IMU2_FILE = R"D:\insv_gyro.dat"

OUTPUT_PARAMS = {
    "imu1_synced": "imu1_synced.csv",
    "imu2_synced": "imu2_synced.csv",
    "plot_time_sync": "time_sync_bspline.png",
    "plot_error_function": "time_delay_error_function.png",
    "plot_rotation_alignment": "gyro_rotation_alignment.png",
    "plot_alignment_error": "rotation_alignment_error.png",
}

# B-spline 参数
SPLINE_KNOTS = 3
SPLINE_SMOOTHING = 0

# ============================================================================
# 数据加载模块
# ============================================================================


def load_imu_data(imu1_path: str, imu2_path: str) -> tuple:
    """
    加载两个IMU的protobuf二进制数据文件

    Args:
        imu1_path: IMU1 .dat文件路径
        imu2_path: IMU2 .dat文件路径

    Returns:
        (t1, t2, gyro_mag1, gyro_mag2): 时间戳和陀螺仪幅度
    """
    print("[数据] 正在加载IMU数据...")

    # 读取protobuf二进制文件
    with open(imu1_path, "rb") as f:
        imu1_msgs = ImuMsgList()
        imu1_msgs.ParseFromString(f.read())

    with open(imu2_path, "rb") as f:
        imu2_msgs = ImuMsgList()
        imu2_msgs.ParseFromString(f.read())

    # 提取数据
    t1 = np.array([msg.timestamp for msg in imu1_msgs.imu_msgs])
    t2 = np.array([msg.timestamp for msg in imu2_msgs.imu_msgs])

    gx1 = np.array([msg.gx for msg in imu1_msgs.imu_msgs])
    gy1 = np.array([msg.gy for msg in imu1_msgs.imu_msgs])
    gz1 = np.array([msg.gz for msg in imu1_msgs.imu_msgs])

    gx2 = np.array([msg.gx for msg in imu2_msgs.imu_msgs])
    gy2 = np.array([msg.gy for msg in imu2_msgs.imu_msgs])
    gz2 = np.array([msg.gz for msg in imu2_msgs.imu_msgs])

    # 计算陀螺仪幅度
    gyro_mag1 = np.sqrt(gx1**2 + gy1**2 + gz1**2)
    gyro_mag2 = np.sqrt(gx2**2 + gy2**2 + gz2**2)

    # todo kk
    t1 -= t1[0]
    t2 -= t2[0]

    print(f"  [IMU1] {len(imu1_msgs.imu_msgs)} 条记录 ({t1[0]:.3f}s - {t1[-1]:.3f}s)")
    print(f"  [IMU2] {len(imu2_msgs.imu_msgs)} 条记录 ({t2[0]:.3f}s - {t2[-1]:.3f}s)")

    # 保存原始数据用于后续导出
    return {
        "t1": t1,
        "t2": t2,
        "gyro_mag1": gyro_mag1,
        "gyro_mag2": gyro_mag2,
        "gx1": gx1,
        "gy1": gy1,
        "gz1": gz1,
        "gx2": gx2,
        "gy2": gy2,
        "gz2": gz2,
        "ax1": np.array([msg.ax for msg in imu1_msgs.imu_msgs]),
        "ay1": np.array([msg.ay for msg in imu1_msgs.imu_msgs]),
        "az1": np.array([msg.az for msg in imu1_msgs.imu_msgs]),
        "ax2": np.array([msg.ax for msg in imu2_msgs.imu_msgs]),
        "ay2": np.array([msg.ay for msg in imu2_msgs.imu_msgs]),
        "az2": np.array([msg.az for msg in imu2_msgs.imu_msgs]),
    }


def align_data_length(
    t1: np.ndarray,
    t2: np.ndarray,
    gyro_mag1: np.ndarray,
    gyro_mag2: np.ndarray,
) -> tuple:
    """
    对齐两个IMU数据的长度到最小值

    Args:
        t1, t2: 时间戳
        gyro_mag1, gyro_mag2: 陀螺仪幅度

    Returns:
        (t1_aligned, t2_aligned, gyro1_aligned, gyro2_aligned, min_len)
    """
    min_len = min(len(gyro_mag1), len(gyro_mag2))

    t1_aligned = t1[:min_len]
    t2_aligned = t2[:min_len]
    gyro1_aligned = gyro_mag1[:min_len]
    gyro2_aligned = gyro_mag2[:min_len]

    return t1_aligned, t2_aligned, gyro1_aligned, gyro2_aligned, min_len


# ============================================================================
# 时间同步模块
# ============================================================================


def create_bspline_models(
    t_aligned: np.ndarray,
    gyro_aligned: np.ndarray,
    k: int = SPLINE_KNOTS,
    s: float = SPLINE_SMOOTHING,
) -> UnivariateSpline:
    """
    为陀螺仪幅度创建B-spline插值模型

    Args:
        t_aligned: 时间戳
        gyro_aligned: 陀螺仪幅度
        k: spline阶数
        s: 平滑因子

    Returns:
        UnivariateSpline 模型
    """
    return UnivariateSpline(t_aligned, gyro_aligned, k=k, s=s)


def compute_cross_correlation(
    t1: np.ndarray,
    t2: np.ndarray,
    gyro1: np.ndarray,
    gyro2: np.ndarray,
    max_delay: float,
) -> tuple:
    """
    计算两个陀螺仪幅度信号的互相关系数

    原理: 1. 将两个信号插值到公共时间轴
          2. 计算插值后信号的互相关
          3. 找到最大相关系数对应的时间延迟

    Args:
        t1, t2: 时间戳数组
        gyro1, gyro2: 陀螺仪幅度数组
        max_delay: 最大评估延迟（秒）

    Returns:
        (time_delays, cross_corr, optimal_delay, max_correlation): 时间延迟、相关系数、最优延迟、最大值
    """
    # 标准化信号
    gyro1_norm = (gyro1 - np.mean(gyro1)) / np.std(gyro1)
    gyro2_norm = (gyro2 - np.mean(gyro2)) / np.std(gyro2)

    # 创建公共时间轴（使用较密集的采样率以保留细节）
    t_min = max(t1[0], t2[0])
    t_max = min(t1[-1], t2[-1])
    dt = min(np.mean(np.diff(t1)), np.mean(np.diff(t2)))
    t_common = np.arange(t_min, t_max, dt)

    # 将两个信号插值到公共时间轴
    # np.interp() 使用线性插值：
    # - 第1参数 t_common: 目标时间点（新采样轴）
    # - 第2参数 t1: 原始时间戳（已知数据的时间位置）
    # - 第3参数 gyro1_norm: 原始陀螺仪值（已知数据的幅度）
    # 返回值：在 t_common 每一时刻的插值结果
    gyro1_interp = np.interp(t_common, t1, gyro1_norm)
    gyro2_interp = np.interp(t_common, t2, gyro2_norm)

    # 计算互相关
    correlation = signal.correlate(gyro1_interp, gyro2_interp, mode="full")
    lags = signal.correlation_lags(len(gyro1_interp), len(gyro2_interp), mode="full")

    # 将lag转换为时间延迟（lag是样本索引差，乘以采样间隔得到时间差）
    time_delays = lags * dt

    # 筛选在max_delay范围内的延迟
    valid_mask = np.abs(time_delays) <= max_delay
    valid_delays = time_delays[valid_mask]
    valid_corr = correlation[valid_mask]

    # 找到最大相关系数
    max_idx = np.argmax(valid_corr)
    optimal_delay = valid_delays[max_idx]
    max_correlation = valid_corr[max_idx]

    return valid_delays, valid_corr, optimal_delay, max_correlation


def find_coarse_alignment(
    t1_aligned: np.ndarray,
    t2_aligned: np.ndarray,
    gyro1_aligned: np.ndarray,
    gyro2_aligned: np.ndarray,
) -> tuple:
    """
    使用互相关系数进行粗对齐

    Args:
        t1_aligned, t2_aligned: 对齐后的时间戳
        gyro1_aligned, gyro2_aligned: 对齐后的陀螺仪幅度

    Returns:
        (optimal_delay, max_correlation, time_delays, correlation): 最优延迟、最大相关系数、时间延迟序列、相关系数序列
    """
    print("[粗对齐] 正在进行粗对齐（基于互相关系数）...")

    # 使用对齐范围内的最大延迟（无上限限制）
    time_range = t1_aligned[-1] - t1_aligned[0]

    time_delays, correlation, optimal_delay, max_correlation = (
        compute_cross_correlation(
            t1_aligned, t2_aligned, gyro1_aligned, gyro2_aligned, time_range
        )
    )

    print(f"  ⏱️  粗对齐延迟: {optimal_delay*1000:.3f} ms")
    print(f"  📈 最大互相关系数: {max_correlation:.6f}")

    return optimal_delay, max_correlation, time_delays, correlation


def plot_coarse_alignment(
    time_delays: np.ndarray,
    correlation: np.ndarray,
    optimal_delay: float,
    max_correlation: float,
) -> None:
    """绘制互相关系数图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        time_delays * 1000, correlation, linewidth=2, color="blue", label="互相关系数"
    )
    ax.axvline(
        optimal_delay * 1000,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"最优延迟: {optimal_delay*1000:.3f}ms",
    )
    ax.scatter(
        [optimal_delay * 1000],
        [max_correlation],
        color="red",
        s=100,
        marker="o",
        zorder=5,
        label=f"最大值: {max_correlation:.6f}",
    )

    ax.set_xlabel("时间延迟 (ms)", fontsize=12)
    ax.set_ylabel("互相关系数", fontsize=12)
    ax.set_title("粗对齐：互相关系数分析", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("coarse_alignment_correlation.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()

    print("  [保存] coarse_alignment_correlation.png")


def plot_coarse_alignment_gyro_comparison(
    t1: np.ndarray,
    t2: np.ndarray,
    gyro_mag1: np.ndarray,
    gyro_mag2: np.ndarray,
    coarse_delay: float,
) -> None:
    """绘制粗对齐后的IMU norm值对比图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制IMU1的原始数据
    ax.plot(
        t1,
        gyro_mag1,
        "g-",
        linewidth=1.5,
        label="IMU1 陀螺仪幅度",
        alpha=0.8,
    )

    # 绘制IMU2的粗对齐后的数据（时间轴加上延迟）
    t2_aligned = t2 + coarse_delay
    ax.plot(
        t2_aligned,
        gyro_mag2,
        "r-",
        linewidth=1.5,
        label=f"IMU2 陀螺仪幅度 (延迟: {coarse_delay*1000:.3f}ms)",
        alpha=0.8,
    )

    # 标注交集范围
    t_overlap_start = max(t1[0], t2_aligned[0])
    t_overlap_end = min(t1[-1], t2_aligned[-1])
    if t_overlap_start < t_overlap_end:
        ax.axvspan(
            t_overlap_start, t_overlap_end, alpha=0.1, color="yellow", label="时间交集"
        )

    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("陀螺仪幅度 (rad/s)", fontsize=12)
    ax.set_title("粗对齐后的IMU norm值对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("coarse_alignment_gyro_comparison.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()

    print("  [保存] coarse_alignment_gyro_comparison.png")


def compute_time_alignment_error(
    tau: float,
    t_common: np.ndarray,
    spl1: UnivariateSpline,
    spl2: UnivariateSpline,
    t2_bounds: tuple,
) -> float:
    """
    计算时间对齐误差（均方误差）

    原理: t2_synced = t2 + tau
          在公共时间轴上比较两个平滑的陀螺仪幅度

    Args:
        tau: 时间延迟（秒）
        t_common: 公共时间轴
        spl1: IMU1的spline模型
        spl2: IMU2的spline模型
        t2_bounds: IMU2时间范围 (t_min, t_max)

    Returns:
        均方误差
    """
    y1 = spl1(t_common)
    t_eval = t_common - tau
    valid_mask = (t_eval >= t2_bounds[0]) & (t_eval <= t2_bounds[1])

    # 要求至少一半的数据有效
    if np.sum(valid_mask) < len(t_common) // 2:
        return 1e10

    y2 = spl2(t_eval[valid_mask])
    y1_valid = y1[valid_mask]

    return np.mean((y1_valid - y2) ** 2)


def optimize_time_delay(
    t_common: np.ndarray,
    spl1: UnivariateSpline,
    spl2: UnivariateSpline,
    t2_bounds: tuple,
    initial_guess: float = None,
) -> float:
    """
    通过优化找到最优时间延迟

    Args:
        t_common: 公共时间轴
        spl1, spl2: spline模型
        t2_bounds: IMU2时间范围
        initial_guess: 初始猜测值（来自粗对齐）

    Returns:
        最优时间延迟（秒）
    """
    print("[精细对齐] 正在进行精细对齐（基于B-spline优化）...")

    # minimize_scalar 使用初始值的方式：
    # - method="bounded": 边界优化，不支持x0参数，直接在bounds范围内搜索
    # - method="brent": Brent方法，可用 bracket 参数指定搜索区间
    # - method="golden": 黄金分割法，可用 bracket 参数指定搜索区间
    #
    # 使用初始值的最佳实践：
    # 当有初始猜测时，用 bracket=(lower, initial_guess, upper) 加速收敛
    # bracket 是三个点 (a, c, b)，其中 a < c < b，c是中点（初始值）

    bracket_lower = initial_guess - 6.0
    bracket_upper = initial_guess + 6.0

    result = minimize_scalar(
        lambda tau: compute_time_alignment_error(tau, t_common, spl1, spl2, t2_bounds),
        method="bounded",
        bounds=(bracket_lower, bracket_upper),
    )

    time_delay = result.x
    error = result.fun

    print(f"  [延迟] 精细对齐延迟: {time_delay*1000:.3f} ms")
    print(f"  [误差] 对齐误差 (MSE): {error:.6f}")

    return time_delay


def sync_imu_timestamps(
    imu1: pd.DataFrame, imu2: pd.DataFrame, time_delay: float
) -> tuple:
    """
    根据计算出的时间延迟同步IMU数据

    Args:
        imu1: IMU1 DataFrame
        imu2: IMU2 DataFrame
        time_delay: 时间延迟（秒）

    Returns:
        (imu1_synced, imu2_synced) 同步后的DataFrame
    """
    print("[同步] 正在同步数据...")

    imu1_synced = imu1.copy()
    imu2_synced = imu2.copy()

    # 调整IMU2的时间戳
    t2_original = imu2["timestamp"].values
    t2_synced = t2_original + time_delay

    imu2_synced.rename(columns={"timestamp": "timestamp_original"}, inplace=True)
    imu2_synced.insert(0, "timestamp", t2_synced)

    return imu1_synced, imu2_synced


def save_synced_data(imu1_synced: pd.DataFrame, imu2_synced: pd.DataFrame) -> None:
    """保存同步后的数据到CSV文件"""
    # 不再保存同步后的数据
    pass
    # imu1_synced.to_csv(OUTPUT_PARAMS["imu1_synced"], index=False)
    # imu2_synced.to_csv(OUTPUT_PARAMS["imu2_synced"], index=False)
    # print(f"  💾 已保存: {OUTPUT_PARAMS['imu1_synced']}")
    # print(f"  💾 已保存: {OUTPUT_PARAMS['imu2_synced']}")


# ============================================================================
# 旋转矩阵计算模块
# ============================================================================


def create_gyro_vector_models(
    t_aligned: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    k: int = SPLINE_KNOTS,
) -> tuple:
    """
    为陀螺仪向量的三个分量创建B-spline模型

    Args:
        t_aligned: 时间戳
        gx, gy, gz: 陀螺仪的三个分量

    Returns:
        (spl_gx, spl_gy, spl_gz) spline模型
    """
    return (
        UnivariateSpline(t_aligned, gx, k=k, s=0),
        UnivariateSpline(t_aligned, gy, k=k, s=0),
        UnivariateSpline(t_aligned, gz, k=k, s=0),
    )


def compute_interpolated_gyro_vectors(
    t_common: np.ndarray,
    t_eval_valid: np.ndarray,
    spl_gx1,
    spl_gy1,
    spl_gz1,
    spl_gx2,
    spl_gy2,
    spl_gz2,
    valid_mask: np.ndarray,
    time_delay: float,
) -> tuple:
    """
    在时间同步后的时间点上插值陀螺仪向量

    Args:
        t_common: 公共时间轴
        t_eval_valid: 有效的评估时间点
        spl_*: 各个分量的spline模型
        valid_mask: 有效数据掩码
        time_delay: 时间延迟

    Returns:
        (gyro1_interp, gyro2_interp) 插值后的陀螺仪向量矩阵
    """
    gyro1_interp = np.column_stack(
        [
            spl_gx1(t_common[valid_mask]),
            spl_gy1(t_common[valid_mask]),
            spl_gz1(t_common[valid_mask]),
        ]
    )

    gyro2_interp = np.column_stack(
        [
            spl_gx2(t_eval_valid - time_delay),
            spl_gy2(t_eval_valid - time_delay),
            spl_gz2(t_eval_valid - time_delay),
        ]
    )

    return gyro1_interp, gyro2_interp


def compute_rotation_matrix_umeyama(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    使用Umeyama算法计算最优旋转矩阵

    原理: 计算旋转矩阵R使得 ||P - R @ Q.T||_F^2 最小
    应用: gyro_i1 = R @ gyro_i2

    Args:
        P: N x 3 矩阵（参考数据，如IMU1的陀螺仪）
        Q: N x 3 矩阵（待旋转数据，如IMU2的陀螺仪）

    Returns:
        3 x 3 旋转矩阵R
    """
    # 计算质心
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # 中心化
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # 计算协方差矩阵
    H = Q_centered.T @ P_centered

    # SVD分解
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 确保det(R) = 1（保证是旋转矩阵而不是反射）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def analyze_rotation(R_matrix: np.ndarray) -> tuple:
    """
    分析旋转矩阵的性质

    Args:
        R_matrix: 3 x 3 旋转矩阵

    Returns:
        (euler_angles, ortho_check, det): 欧拉角、正交性检验矩阵、行列式
    """
    # 计算欧拉角
    rotation = Rotation.from_matrix(R_matrix)
    euler_angles = rotation.as_euler("xyz", degrees=True)

    # 验证正交性
    ortho_check = R_matrix @ R_matrix.T

    # 计算行列式
    det_R = np.linalg.det(R_matrix)

    return euler_angles, ortho_check, det_R


def compute_alignment_metrics(
    gyro1_interp: np.ndarray,
    gyro2_interp: np.ndarray,
    R_matrix: np.ndarray,
) -> tuple:
    """
    计算对齐质量指标

    Args:
        gyro1_interp: IMU1插值的陀螺仪向量
        gyro2_interp: IMU2插值的陀螺仪向量
        R_matrix: 旋转矩阵

    Returns:
        (gyro1_rotated, alignment_error, errors_per_axis)
    """
    # 旋转IMU2数据
    gyro1_rotated = (R_matrix @ gyro2_interp.T).T

    # 计算对齐误差
    errors_per_axis = np.linalg.norm(gyro1_interp - gyro1_rotated, axis=1)
    alignment_error = np.mean(errors_per_axis)

    return gyro1_rotated, alignment_error, errors_per_axis


# ============================================================================
# 可视化模块
# ============================================================================


def plot_time_sync_result(
    t1_aligned: np.ndarray,
    t2_aligned: np.ndarray,
    t_common: np.ndarray,
    gyro1_aligned: np.ndarray,
    gyro2_aligned: np.ndarray,
    gyro_magnitude2: np.ndarray,
    spl1: UnivariateSpline,
    spl2: UnivariateSpline,
    gyro1_smooth: np.ndarray,
    gyro2_smooth: np.ndarray,
    time_delay: float,
) -> None:
    """绘制时间同步的B-spline对比"""
    print("[图表] 正在绘制B-spline对比图...")

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        t1_aligned,
        gyro1_aligned,
        "o",
        label="IMU1 原始数据",
        linewidth=0.5,
        markersize=2,
        alpha=0.5,
        color="green",
    )
    ax.plot(
        t_common,
        gyro1_smooth,
        "-",
        label="IMU1 B样条插值",
        linewidth=2,
        color="green",
    )
    ax.plot(
        t2_aligned,
        gyro2_aligned,
        "o",
        label="IMU2 原始数据(同步前)",
        linewidth=0.5,
        markersize=2,
        alpha=0.5,
        color="red",
    )
    ax.plot(
        t_common,
        gyro2_smooth,
        "-",
        label="IMU2 B样条插值(同步前)",
        linewidth=1.5,
        color="red",
    )

    # 同步后的数据
    t2_synced = t2_aligned + time_delay
    ax.scatter(
        t2_synced,
        gyro_magnitude2,
        c="orange",
        s=5,
        alpha=1,
        label="IMU2 同步数据",
    )
    ax.plot(
        t_common,
        spl2(t_common - time_delay),
        "-",
        label="IMU2 B样条(同步后)",
        linewidth=2,
        color="orange",
    )

    ax.set_xlabel("GPS时间 (s)", fontsize=12)
    ax.set_ylabel("陀螺仪幅度 (rad/s)", fontsize=12)
    ax.set_title(
        f"B样条插值对比：IMU1 vs IMU2 (延迟: {time_delay*1000:.3f}ms)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PARAMS["plot_time_sync"], dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()

    print(f"  [保存] {OUTPUT_PARAMS['plot_time_sync']}")


def plot_time_delay_error_function(
    time_delay: float,
    t_common: np.ndarray,
    spl1: UnivariateSpline,
    spl2: UnivariateSpline,
    t2_bounds: tuple,
) -> None:
    """绘制时间延迟优化函数"""
    print("[图表] 正在绘制误差函数图...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # 计算误差函数
    tau_range = np.linspace(time_delay - 0.05, time_delay + 0.05, 50)
    errors = [
        compute_time_alignment_error(tau, t_common, spl1, spl2, t2_bounds)
        for tau in tau_range
    ]

    ax.plot(tau_range * 1000, errors, "g-", linewidth=3, label="对齐误差")
    ax.axvline(
        x=time_delay * 1000,
        color="r",
        linestyle="--",
        linewidth=3,
        label=f"最优延迟: {time_delay*1000:.3f}ms",
    )

    ax.set_xlabel("时间延迟 (ms)", fontsize=12)
    ax.set_ylabel("MSE 误差", fontsize=12)
    ax.set_title("连续时间优化：误差函数", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PARAMS["plot_error_function"], dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()

    print(f"  [保存] {OUTPUT_PARAMS['plot_error_function']}")


def plot_rotation_alignment(
    t_common_sync: np.ndarray,
    t_eval_valid: np.ndarray,
    valid_mask: np.ndarray,
    gyro1_interp: np.ndarray,
    gyro2_interp: np.ndarray,
    gyro1_rotated: np.ndarray,
    euler_angles: np.ndarray,
) -> None:
    """绘制旋转对齐的陀螺仪数据对比"""
    print("[图表] 正在绘制旋转对齐对比图...")

    fig = plt.figure(figsize=(16, 10))

    axes = [fig.add_subplot(3, 2, i) for i in range(1, 7)]

    # 使用同步后的公共时间轴对应的有效部分
    t_common_valid = t_common_sync[valid_mask]

    # 原始数据对比（未旋转）
    axis_labels = ["X轴", "Y轴", "Z轴"]
    for idx in range(3):
        ax = axes[idx]
        ax.plot(
            t_common_valid,
            gyro1_interp[:, idx],
            "g-",
            linewidth=2,
            label=f"IMU1 g{chr(120+idx)}",
        )
        ax.plot(
            t_common_valid,
            gyro2_interp[:, idx],
            "r--",
            linewidth=1.5,
            label=f"IMU2 g{chr(120+idx)} (同步前旋转)",
        )
        ax.set_xlabel("时间 (s)", fontsize=10)
        ax.set_ylabel("角速度 (rad/s)", fontsize=10)
        ax.set_title(
            f"{axis_labels[idx]}陀螺仪对比（旋转前）", fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 旋转后数据对比
    for idx in range(3):
        ax = axes[idx + 3]
        ax.plot(
            t_common_valid,
            gyro1_interp[:, idx],
            "g-",
            linewidth=2,
            label=f"IMU1 g{chr(120+idx)}",
        )
        ax.plot(
            t_common_valid,
            gyro1_rotated[:, idx],
            "b-",
            linewidth=2,
            label=f"IMU2 g{chr(120+idx)} (旋转后)",
        )
        ax.set_xlabel("时间 (s)", fontsize=10)
        ax.set_ylabel("角速度 (rad/s)", fontsize=10)
        ax.set_title(
            f"{axis_labels[idx]}陀螺仪对比（旋转后）", fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"陀螺仪旋转对齐效果对比 (yaw={euler_angles[2]:.2f}°, pitch={euler_angles[1]:.2f}°, roll={euler_angles[0]:.2f}°)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PARAMS["plot_rotation_alignment"], dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()

    print(f"  [保存] {OUTPUT_PARAMS['plot_rotation_alignment']}")


def plot_alignment_error_analysis(
    t_common_sync: np.ndarray,
    valid_mask: np.ndarray,
    errors_per_axis: np.ndarray,
    alignment_error: float,
) -> None:
    """绘制对齐误差分析"""
    print("[图表] 正在绘制误差分析图...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 使用同步后的公共时间轴对应的有效部分
    t_common_valid = t_common_sync[valid_mask]

    # 误差分布直方图
    ax1.hist(errors_per_axis, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.axvline(
        x=alignment_error,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"平均误差: {alignment_error:.6f}",
    )
    ax1.set_xlabel("误差 (rad/s)", fontsize=11)
    ax1.set_ylabel("频数", fontsize=11)
    ax1.set_title("对齐误差分布", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # 误差时间序列
    ax2.plot(
        t_common_valid,
        errors_per_axis,
        "o-",
        color="purple",
        markersize=3,
        linewidth=1,
        alpha=0.7,
    )
    ax2.axhline(
        y=alignment_error,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"平均误差: {alignment_error:.6f}",
    )
    ax2.set_xlabel("时间 (s)", fontsize=11)
    ax2.set_ylabel("误差 (rad/s)", fontsize=11)
    ax2.set_title("对齐误差时间序列", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("旋转对齐误差分析", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PARAMS["plot_alignment_error"], dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()

    print(f"  [保存] {OUTPUT_PARAMS['plot_alignment_error']}")


# ============================================================================
# 结果输出模块
# ============================================================================


def print_time_sync_results(time_delay: float) -> None:
    """打印时间同步结果"""
    print("\n" + "=" * 60)
    print("[结果] 时间同步结果")
    print("=" * 60)
    print(f"  ⏱️  时间延迟: {time_delay*1000:.3f} ms ({time_delay:.6f} s)")
    print("  [说明] IMU2 的时间戳应加上此延迟与 IMU1 对齐")


def print_rotation_results(
    R_matrix: np.ndarray,
    euler_angles: np.ndarray,
    alignment_error: float,
    ortho_check: np.ndarray,
    det_R: float,
) -> None:
    """打印旋转矩阵和对齐结果"""
    print("\n" + "=" * 60)
    print("[旋转] 旋转矩阵和对齐结果")
    print("=" * 60)
    print("\n  [矩阵] 相对旋转矩阵 R (gyro_i1 = R @ gyro_i2):")
    print(R_matrix)
    print(f"\n  🔄 欧拉角 (XYZ顺序，单位: 度):")
    print(f"    ↻ roll  (X轴旋转): {euler_angles[0]:7.3f}°")
    print(f"    ↻ pitch (Y轴旋转): {euler_angles[1]:7.3f}°")
    print(f"    ↻ yaw   (Z轴旋转): {euler_angles[2]:7.3f}°")
    print(f"\n  ❌ 对齐误差 (3D向量均方根): {alignment_error:.6f} rad/s")
    print("\n  [验证] 正交性验证 (R @ R^T = I):")
    print(ortho_check)
    print(f"  🔍 det(R) = {det_R:.6f} (应为 ±1)")

    if abs(det_R - 1.0) < 1e-6:
        print("  [OK] 旋转矩阵有效")
    else:
        print("  [警告] det(R) 偏离 1")


def print_summary() -> None:
    """打印处理完成总结"""
    print("\n" + "=" * 60)
    print("[完成] 处理完成")
    print("=" * 60)
    print("  [图表] 已生成图表:")
    print(f"    📈 coarse_alignment_correlation.png")
    print(f"    📊 coarse_alignment_gyro_comparison.png")
    print(f"    📈 {OUTPUT_PARAMS['plot_time_sync']}")
    print(f"    📉 {OUTPUT_PARAMS['plot_error_function']}")
    print(f"    🔄 {OUTPUT_PARAMS['plot_rotation_alignment']}")
    print(f"    📊 {OUTPUT_PARAMS['plot_alignment_error']}")


# ============================================================================
# 主程序
# ============================================================================


def main():
    """主程序入口"""
    print("\n" + "=" * 60)
    print("[开始] IMU 时间同步和旋转对齐分析")
    print("=" * 60)

    # ===== 第1步: 加载数据 =====
    data = load_imu_data(IMU1_FILE, IMU2_FILE)
    t1, t2 = data["t1"], data["t2"]
    gyro_mag1, gyro_mag2 = data["gyro_mag1"], data["gyro_mag2"]

    # 为了兼容后续的 sync_imu_timestamps，创建DataFrame
    imu1 = pd.DataFrame(
        {
            "timestamp": t1,
            "ax": data["ax1"],
            "ay": data["ay1"],
            "az": data["az1"],
            "gx": data["gx1"],
            "gy": data["gy1"],
            "gz": data["gz1"],
        }
    )
    imu2 = pd.DataFrame(
        {
            "timestamp": t2,
            "ax": data["ax2"],
            "ay": data["ay2"],
            "az": data["az2"],
            "gx": data["gx2"],
            "gy": data["gy2"],
            "gz": data["gz2"],
        }
    )

    # ===== 第2步: 粗对齐（基于互相关系数）=====
    coarse_delay, max_corr, time_delays, correlation = find_coarse_alignment(
        t1, t2, gyro_mag1, gyro_mag2
    )
    plot_coarse_alignment(time_delays, correlation, coarse_delay, max_corr)
    plot_coarse_alignment_gyro_comparison(t1, t2, gyro_mag1, gyro_mag2, coarse_delay)

    # ===== 第2.5步: 计算粗对齐后的公共时间轴 =====
    # 粗对齐后，IMU2的时间轴变为 t2 + coarse_delay
    t2_aligned = t2 + coarse_delay
    t_common_start = max(t1[0], t2_aligned[0])
    t_common_end = min(t1[-1], t2_aligned[-1])
    
    # 确保交集有效
    if t_common_start >= t_common_end:
        print("[错误] 粗对齐后时间轴无交集，无法继续处理！")
        return
    
    # 使用较密集的采样率
    dt1 = np.mean(np.diff(t1))
    dt2 = np.mean(np.diff(t2))
    dt_common = min(dt1, dt2)
    num_samples = int((t_common_end - t_common_start) / dt_common) + 1
    t_common = np.linspace(t_common_start, t_common_end, num_samples)
    
    print(f"[时间轴] 公共时间范围: [{t_common_start:.3f}, {t_common_end:.3f}] ({num_samples} 个采样点)")

    # ===== 第3步: 创建B-spline模型 =====
    print("[模型] 正在创建B-spline插值模型...")
    spl1 = create_bspline_models(t1, gyro_mag1)
    spl2 = create_bspline_models(t2, gyro_mag2)

    gyro1_smooth = spl1(t_common)
    gyro2_smooth = spl2(t_common - coarse_delay)

    # ===== 第4步: 精细对齐（基于B-spline优化）=====
    time_delay = optimize_time_delay(
        t_common,
        spl1,
        spl2,
        (t2[0], t2[-1]),
        initial_guess=coarse_delay,
    )

    # ===== 第5步: 同步IMU数据 =====
    imu1_synced, imu2_synced = sync_imu_timestamps(imu1, imu2, time_delay)
    save_synced_data(imu1_synced, imu2_synced)

    # ===== 第5.5步: 基于最终的time_delay，计算有效的数据掩码 =====
    # t_common已经是粗对齐后的公共时间轴
    # 现在需要考虑最终的time_delay与coarse_delay的差异
    # 计算在这个t_common范围内，对应的IMU2时间是否有效
    t_eval = t_common - time_delay
    valid_mask = (t_eval >= t2[0]) & (t_eval <= t2[-1])
    
    # 确保至少有一半的数据有效
    if np.sum(valid_mask) < len(t_common) // 2:
        print("[警告] 最终对齐后有效数据不足！")
        return
    
    # 提取有效部分
    t_common_valid = t_common[valid_mask]
    t_eval_valid = t_eval[valid_mask]
    
    print(f"[数据] 有效样本数: {np.sum(valid_mask)} / {len(t_common)} ({100*np.sum(valid_mask)/len(t_common):.1f}%)")

    # ===== 第6步: 提取陀螺仪向量数据 =======
    print("正在处理陀螺仪向量数据...")
    gx1, gy1, gz1 = (
        imu1["gx"].values,
        imu1["gy"].values,
        imu1["gz"].values,
    )
    gx2, gy2, gz2 = (
        imu2["gx"].values,
        imu2["gy"].values,
        imu2["gz"].values,
    )

    # ===== 第7步: 创建陀螺仪向量的B-spline模型 =====
    spl_gx1, spl_gy1, spl_gz1 = create_gyro_vector_models(t1, gx1, gy1, gz1)
    spl_gx2, spl_gy2, spl_gz2 = create_gyro_vector_models(t2, gx2, gy2, gz2)

    # ===== 第8步: 计算插值的陀螺仪向量 =====
    gyro1_interp, gyro2_interp = compute_interpolated_gyro_vectors(
        t_common,
        t_eval_valid,
        spl_gx1,
        spl_gy1,
        spl_gz1,
        spl_gx2,
        spl_gy2,
        spl_gz2,
        valid_mask,
        time_delay,
    )

    # ===== 第9步: 计算旋转矩阵 =====
    print("正在计算旋转矩阵...")
    R_matrix = compute_rotation_matrix_umeyama(gyro1_interp, gyro2_interp)

    # ===== 第10步: 分析旋转矩阵 =====
    euler_angles, ortho_check, det_R = analyze_rotation(R_matrix)

    # ===== 第11步: 计算对齐误差 =====
    gyro1_rotated, alignment_error, errors_per_axis = compute_alignment_metrics(
        gyro1_interp, gyro2_interp, R_matrix
    )

    # ===== 第12步: 打印结果 =====
    print_time_sync_results(time_delay)
    print_rotation_results(R_matrix, euler_angles, alignment_error, ortho_check, det_R)

    # ===== 第13步: 绘制图表 =====
    print("\n正在生成可视化图表...")
    plot_time_sync_result(
        t1,
        t2,
        t_common,
        gyro_mag1,
        gyro_mag2,
        gyro_mag2,
        spl1,
        spl2,
        gyro1_smooth,
        gyro2_smooth,
        time_delay,
    )
    plot_time_delay_error_function(time_delay, t_common, spl1, spl2, (t2[0], t2[-1]))
    plot_rotation_alignment(
        t_common,
        t_eval_valid,
        valid_mask,
        gyro1_interp,
        gyro2_interp,
        gyro1_rotated,
        euler_angles,
    )
    plot_alignment_error_analysis(
        t_common, valid_mask, errors_per_axis, alignment_error
    )

    # ===== 第14步: 打印总结 =====
    print_summary()


if __name__ == "__main__":
    main()
