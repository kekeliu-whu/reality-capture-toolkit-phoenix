"""
IMU time synchronization tool.

This module:
1. Loads data from two IMU sensors.
2. Synchronizes time using B-spline interpolation over gyro magnitude.
3. Generates visualization plots for analysis.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy import signal
import matplotlib
import argparse
from proto.sensors_pb2 import ImuMsgList

from spdlog_compat import init_spdlog_like_logger


LOGGER = init_spdlog_like_logger()

# ============================================================================
# [CONFIG]  配置参数 - 修改此处来调整程序行为
# ============================================================================

# ========== 显示设置 ==========
SHOW_PLOTS = False  # 是否弹出显示图形（False=仅保存，True=显示）
SAVE_PLOTS = False  # 是否保存图形到文件（coarse_alignment_correlation.png, fine_alignment_gyro_comparison.png）

# ========== 文件路径（以下通过命令行参数传入，此处为默认值）==========
DEFAULT_IMU_DEVICE_FILE = R"D:/output/imu.dat"
DEFAULT_IMU_INSTA_FILE = R"D:/output/images/insv.dat"

# ========== B-spline 插值参数 ==========
SPLINE_KNOTS = 3  # B-spline 的阶数（次数）
SPLINE_SMOOTHING = 0  # 平滑因子 (0=精确插值，>0=平滑)

# ========== 时间对齐优化参数 ==========
OPTIMIZATION_SEARCH_RANGE = 1.0  # 精细对齐的搜索范围 (±秒)

# ========== 数据清理参数 ==========
TIME_TOLERANCE = 1e-9  # 时间戳最小差异阈值，用于移除重复点

# ============================================================================
# matplotlib 配置（需要在导入pyplot前设置）
# ============================================================================

if not SHOW_PLOTS:
    matplotlib.use("Agg")  # 使用非GUI后端，禁用窗口弹出

import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = True

# ============================================================================
# 数据加载模块
# ============================================================================


def load_imu_data(imu_device_path: str, imu_insta_path: str) -> tuple:
    """
    加载两个IMU的protobuf二进制数据文件

    Args:
        imu_device_path: Device .dat文件路径
        imu_insta_path: Insta .dat文件路径

    Returns:
        (t1, t2, gyro_mag1, gyro_mag2): 时间戳和陀螺仪幅度
    """
    print("[INFO] Loading IMU data...")

    # 读取protobuf二进制文件
    with open(imu_device_path, "rb") as f:
        imu_device_msgs = ImuMsgList()
        imu_device_msgs.ParseFromString(f.read())

    with open(imu_insta_path, "rb") as f:
        imu_insta_msgs = ImuMsgList()
        imu_insta_msgs.ParseFromString(f.read())

    # 提取数据
    t_device = np.array([msg.timestamp for msg in imu_device_msgs.imu_msgs])
    t_insta = np.array([msg.timestamp for msg in imu_insta_msgs.imu_msgs])

    gx_device = np.array([msg.gx for msg in imu_device_msgs.imu_msgs])
    gy_device = np.array([msg.gy for msg in imu_device_msgs.imu_msgs])
    gz_device = np.array([msg.gz for msg in imu_device_msgs.imu_msgs])

    gx_insta = np.array([msg.gx for msg in imu_insta_msgs.imu_msgs])
    gy_insta = np.array([msg.gy for msg in imu_insta_msgs.imu_msgs])
    gz_insta = np.array([msg.gz for msg in imu_insta_msgs.imu_msgs])

    # 计算陀螺仪幅度
    gyro_mag_device = np.sqrt(gx_device**2 + gy_device**2 + gz_device**2)
    gyro_mag_insta = np.sqrt(gx_insta**2 + gy_insta**2 + gz_insta**2)

    offset = t_insta[0] - t_device[0]
    print(
        f"  Initial Insta-to-Device time offset: {offset:.3f} s, t_device[0]: {t_device[0]:.3f} t_device[end]: {t_device[-1]:.3f}, t_insta[0]: {t_insta[0]:.3f}, t_insta[end]: {t_insta[-1]:.3f}"
    )
    t_device -= t_device[0]
    t_insta -= t_insta[0]

    print(
        f"  [Device] {len(imu_device_msgs.imu_msgs)} samples ({t_device[0]:.3f}s - {t_device[-1]:.3f}s)"
    )
    print(
        f"  [Insta] {len(imu_insta_msgs.imu_msgs)} samples ({t_insta[0]:.3f}s - {t_insta[-1]:.3f}s)"
    )

    # 保存原始数据用于后续导出
    return {
        "t_device": t_device,
        "t_insta": t_insta,
        "gyro_mag_device": gyro_mag_device,
        "gyro_mag_insta": gyro_mag_insta,
        "gx_device": gx_device,
        "gy_device": gy_device,
        "gz_device": gz_device,
        "gx_insta": gx_insta,
        "gy_insta": gy_insta,
        "gz_insta": gz_insta,
        "ax_device": np.array([msg.ax for msg in imu_device_msgs.imu_msgs]),
        "ay_device": np.array([msg.ay for msg in imu_device_msgs.imu_msgs]),
        "az_device": np.array([msg.az for msg in imu_device_msgs.imu_msgs]),
        "ax_insta": np.array([msg.ax for msg in imu_insta_msgs.imu_msgs]),
        "ay_insta": np.array([msg.ay for msg in imu_insta_msgs.imu_msgs]),
        "az_insta": np.array([msg.az for msg in imu_insta_msgs.imu_msgs]),
        "offset": offset,
    }


# ============================================================================
# 时间同步模块
# ============================================================================


def remove_duplicate_times(
    t: np.ndarray,
    y: np.ndarray,
    time_tolerance: float = TIME_TOLERANCE,
) -> tuple:
    """
    移除时间数组中的重复值和非严格递增的点

    当多个数据点具有相同或非常接近的时间戳时，
    UnivariateSpline（当s=0时）会报错"x must be strictly increasing"

    Args:
        t: 时间戳数组
        y: 对应的数据值数组
        time_tolerance: 时间差的最小阈值（用于处理浮点数精度）

    Returns:
        (t_unique, y_unique): 移除重复值后的时间戳和数据值
    """
    if len(t) == 0:
        return t, y

    # 使用更严格的条件：确保严格递增，处理浮点数精度问题
    diffs = np.diff(t)
    unique_mask = np.concatenate(([True], diffs > time_tolerance))

    t_unique = t[unique_mask]
    y_unique = y[unique_mask]

    n_removed = len(t) - len(t_unique)

    # 再次检查：确保没有任何非递增的对
    if len(t_unique) > 1:
        final_diffs = np.diff(t_unique)
        if np.any(final_diffs <= 0):
            # 如果仍然有问题，采用更激进的方法
            print("[WARN] Non-increasing timestamps remain after cleanup, applying extra filtering...")
            # 逐个检查，跳过所有不满足条件的点
            indices = [0]
            for i in range(1, len(t_unique)):
                if t_unique[i] > t_unique[indices[-1]]:
                    indices.append(i)
            t_unique = t_unique[indices]
            y_unique = y_unique[indices]
            n_removed = len(t) - len(t_unique)

    if n_removed > 0:
        print(
            f"  [Cleanup] Removed {n_removed} duplicate or out-of-order timestamps ({len(t_unique)} kept)"
        )

    return t_unique, y_unique


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
    # 移除重复的时间值以确保严格递增
    t_clean, gyro_clean = remove_duplicate_times(t_aligned, gyro_aligned)

    # 检查清理后的数据是否满足最小要求
    if len(t_clean) < k + 2:
        print(f"[WARN] Not enough data points ({len(t_clean)} < {k+2}), falling back to a lower spline order")
        k_adjusted = max(1, len(t_clean) - 2)
        return UnivariateSpline(t_clean, gyro_clean, k=k_adjusted, s=0.01)

    # 尝试使用指定的平滑因子
    try:
        return UnivariateSpline(t_clean, gyro_clean, k=k, s=s)
    except ValueError as e:
        error_str = str(e)
        if "strictly increasing" in error_str or "increasing" in error_str:
            # 如果时间戳仍然有问题，使用更激进的平滑
            print("[INFO] Timestamp issue detected, retrying with higher smoothing")
            try:
                return UnivariateSpline(t_clean, gyro_clean, k=k, s=0.1)
            except ValueError:
                # 如果还是失败，使用最高平滑
                print("[INFO] Retrying with maximum smoothing to handle timestamp issues")
                return UnivariateSpline(t_clean, gyro_clean, k=k, s=1.0)
        else:
            raise


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
    print("[INFO] Running coarse alignment using cross-correlation...")

    # 使用对齐范围内的最大延迟（无上限限制）
    time_range = t1_aligned[-1] - t1_aligned[0]

    time_delays, correlation, optimal_delay, max_correlation = (
        compute_cross_correlation(
            t1_aligned, t2_aligned, gyro1_aligned, gyro2_aligned, time_range
        )
    )

    print(f"  [TIME]  Coarse alignment delay: {optimal_delay*1000:.3f} ms")
    print(f"  [CORR] Maximum cross-correlation: {max_correlation:.6f}")

    return optimal_delay, max_correlation, time_delays, correlation


def find_top_peaks(
    time_delays: np.ndarray,
    correlation: np.ndarray,
    num_peaks: int = 3,
    min_distance: float = 0.05,
) -> list:
    """
    找到前N个最高的互相关系数峰值

    Args:
        time_delays: 时间延迟数组
        correlation: 互相关系数数组
        num_peaks: 要找的峰值数量
        min_distance: 峰值之间的最小距离（秒）

    Returns:
        [(delay, corr), ...] 峰值列表，按相关系数从高到低排序
    """
    # 使用signal.find_peaks找到所有峰值
    min_distance_samples = int(min_distance / np.mean(np.diff(time_delays)))
    peaks, properties = signal.find_peaks(
        correlation,
        distance=max(1, min_distance_samples),
        prominence=np.max(correlation) * 0.05,  # 峰值需要有显著的高度差
    )

    # 获取峰值的延迟和相关系数
    peak_data = [(time_delays[p], correlation[p]) for p in peaks]
    # 按相关系数从高到低排序
    peak_data.sort(key=lambda x: x[1], reverse=True)
    # 返回前N个
    top_peaks = peak_data[:num_peaks]

    print(f"  [Peaks] Found {len(peak_data)} peaks, top {len(top_peaks)} are:")
    for i, (delay, corr) in enumerate(top_peaks):
        print(f"    [{i+1}] Delay: {delay*1000:8.3f} ms, correlation: {corr:.6f}")

    return top_peaks


def plot_coarse_alignment(
    time_delays: np.ndarray,
    correlation: np.ndarray,
    optimal_delay: float,
    max_correlation: float,
    top_peaks: list = None,
) -> None:
    """绘制互相关系数图，包含前N个峰值"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        time_delays * 1000, correlation, linewidth=2, color="blue", label="Cross-correlation"
    )

    # 绘制主要峰值（红色）
    ax.axvline(
        optimal_delay * 1000,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best delay: {optimal_delay*1000:.3f}ms",
    )
    ax.scatter(
        [optimal_delay * 1000],
        [max_correlation],
        color="red",
        s=100,
        marker="o",
        zorder=5,
        label=f"Peak value: {max_correlation:.6f}",
    )

    # 绘制其他候选峰值（如果有）
    if top_peaks and len(top_peaks) > 1:
        for i, (delay, corr) in enumerate(top_peaks[1:]):
            ax.scatter(
                [delay * 1000],
                [corr],
                color="orange",
                s=80,
                marker="s",
                alpha=0.7,
                zorder=4,
            )
            ax.axvline(delay * 1000, color="orange", linestyle=":", alpha=0.5)

    ax.set_xlabel("Time delay (ms)", fontsize=12)
    ax.set_ylabel("Cross-correlation", fontsize=12)
    ax.set_title(
        "Coarse alignment: cross-correlation analysis (top 3 peaks)", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig("coarse_alignment_correlation.png", dpi=150, bbox_inches="tight")
        print("[INFO] Saved coarse_alignment_correlation.png")
    if SHOW_PLOTS:
        plt.show()


def compute_time_alignment_error(
    tau: float,
    t_common: np.ndarray,
    spl_device: UnivariateSpline,
    spl_insta: UnivariateSpline,
    t_insta_bounds: tuple,
) -> float:
    """
    计算时间对齐误差（均方误差）

    原理: t_insta_synced = t_insta + tau
          在公共时间轴上比较两个平滑的陀螺仪幅度

    Args:
        tau: 时间延迟（秒）
        t_common: 公共时间轴
        spl_device: Device的spline模型
        spl_insta: Insta的spline模型
        t_insta_bounds: Insta时间范围 (t_min, t_max)

    Returns:
        均方误差
    """
    y_device = spl_device(t_common)
    t_eval = t_common - tau
    valid_mask = (t_eval >= t_insta_bounds[0]) & (t_eval <= t_insta_bounds[1])

    # 要求至少一半的数据有效
    if np.sum(valid_mask) < len(t_common) // 2:
        return 1e10

    y_insta = spl_insta(t_eval[valid_mask])
    y_device_valid = y_device[valid_mask]

    return np.mean((y_device_valid - y_insta) ** 2)


def optimize_time_delay(
    t_device: np.ndarray,
    t_insta: np.ndarray,
    gyro_mag_device: np.ndarray,
    gyro_mag_insta: np.ndarray,
    t_common: np.ndarray,
    t_insta_bounds: tuple,
    initial_guess: float = None,
    verbose: bool = True,
) -> tuple:
    """
    通过优化找到最优时间延迟

    Args:
        t_device, t_insta: 时间戳数组
        gyro_mag_device, gyro_mag_insta: 陀螺仪幅度数组
        t_common: 公共时间轴
        t_insta_bounds: Insta时间范围
        initial_guess: 初始猜测值（来自粗对齐）
        verbose: 是否打印详细信息

    Returns:
        (最优时间延迟, 对齐误差) 元组
    """
    if verbose:
        print("[INFO] Running fine alignment using B-spline optimization...")

    spl_device = create_bspline_models(t_device, gyro_mag_device)
    spl_insta = create_bspline_models(t_insta, gyro_mag_insta)

    bracket_lower = initial_guess - OPTIMIZATION_SEARCH_RANGE
    bracket_upper = initial_guess + OPTIMIZATION_SEARCH_RANGE

    result = minimize_scalar(
        lambda tau: compute_time_alignment_error(
            tau, t_common, spl_device, spl_insta, t_insta_bounds
        ),
        method="bounded",
        bounds=(bracket_lower, bracket_upper),
    )

    time_delay = result.x
    error = result.fun

    if verbose:
        print(f"  [TIME] Fine alignment delay: {time_delay*1000:.3f} ms")
        print(f"  [MSE] Alignment error: {error:.6f}")

    return time_delay, error


# ============================================================================
# 可视化模块
# ============================================================================


def plot_fine_alignment_gyro_comparison(
    t_device: np.ndarray,
    t_insta: np.ndarray,
    gyro_mag_device: np.ndarray,
    gyro_mag_insta: np.ndarray,
    time_delay: float,
    coarse_delay: float,
) -> None:
    """绘制粗对齐和精细对齐的对比图"""
    print("[INFO] Plotting coarse vs fine alignment comparison...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制Device的原始数据
    ax.plot(
        t_device,
        gyro_mag_device,
        "g-",
        linewidth=2,
        label="Device gyro magnitude (reference)",
        alpha=0.9,
    )

    # 绘制Insta粗对齐后的数据
    t_insta_coarse = t_insta + coarse_delay
    ax.plot(
        t_insta_coarse,
        gyro_mag_insta,
        "orange",
        linewidth=1.5,
        linestyle="--",
        label=f"Insta coarse alignment (delay: {coarse_delay*1000:.3f}ms)",
        alpha=0.7,
    )

    # 绘制Insta精细对齐后的数据（时间轴加上总延迟）
    t_insta_fine = t_insta + time_delay
    ax.plot(
        t_insta_fine,
        gyro_mag_insta,
        "r-",
        linewidth=2,
        label=f"Insta fine alignment (delay: {time_delay*1000:.3f}ms)",
        alpha=0.9,
    )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Gyro magnitude (rad/s)", fontsize=12)
    ax.set_title("Coarse vs fine alignment", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig("fine_alignment_gyro_comparison.png", dpi=150, bbox_inches="tight")
        print("[INFO] Saved fine_alignment_gyro_comparison.png")
    if SHOW_PLOTS:
        plt.show()


# ============================================================================
# 结果输出模块
# ============================================================================


def print_time_sync_results(time_delay: float) -> None:
    """打印时间同步结果"""
    print("\n" + "=" * 60)
    print("[INFO] Time synchronization result")
    print("=" * 60)
    print(f"  [TIME]  Delay: {time_delay*1000:.3f} ms ({time_delay:.6f} s)")
    print("  Insta timestamps should be shifted by this delay to align with Device")


# ============================================================================
# 主程序
# ============================================================================


def main():
    """主程序入口"""
    # ===== 命令行参数解析 =====
    parser = argparse.ArgumentParser(
        description="IMU time synchronization using B-spline interpolation over gyro magnitude"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_IMU_DEVICE_FILE,
        help=f"Path to the Device IMU data file (default: {DEFAULT_IMU_DEVICE_FILE})",
    )
    parser.add_argument(
        "--insta",
        type=str,
        default=DEFAULT_IMU_INSTA_FILE,
        help=f"Path to the Insta IMU data file (default: {DEFAULT_IMU_INSTA_FILE})",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("[INFO] Starting IMU time synchronization and alignment analysis")
    print("=" * 60)

    # ===== 第1步: 加载数据 =====
    data = load_imu_data(args.device, args.insta)
    t_device, t_insta = data["t_device"], data["t_insta"]
    gyro_mag_device, gyro_mag_insta = data["gyro_mag_device"], data["gyro_mag_insta"]

    # ===== 第1.5步: 清理数据（移除重复的时间戳）=====
    print("[INFO] Cleaning raw timestamps...")
    t_device, gyro_mag_device = remove_duplicate_times(t_device, gyro_mag_device)
    t_insta, gyro_mag_insta = remove_duplicate_times(t_insta, gyro_mag_insta)

    # ===== 第2步: 粗对齐（基于互相关系数）=====
    coarse_delay, max_corr, time_delays, correlation = find_coarse_alignment(
        t_device, t_insta, gyro_mag_device, gyro_mag_insta
    )

    # ===== 找到前3个峰值 =====
    top_peaks = find_top_peaks(time_delays, correlation, num_peaks=3)
    plot_coarse_alignment(time_delays, correlation, coarse_delay, max_corr, top_peaks)

    # ===== 第2.5步: 对每个峰值进行粗对齐和精细对齐 =====
    results = []  # 存储 (peak_idx, coarse_delay, fine_delay, error)

    for peak_idx, (coarse_delay_candidate, peak_corr) in enumerate(top_peaks):
        print("\n" + "=" * 60)
        print(
            f"[INFO] Trial {peak_idx+1}/3 using peak {peak_idx+1}: {coarse_delay_candidate*1000:.3f}ms"
        )
        print("=" * 60)

        # 计算粗对齐后的公共时间轴
        t_insta_aligned = t_insta + coarse_delay_candidate
        t_common_start = max(t_device[0], t_insta_aligned[0])
        t_common_end = min(t_device[-1], t_insta_aligned[-1])

        # 确保交集有效
        if t_common_start >= t_common_end:
            print("[ERROR] No overlapping time range after coarse alignment, skipping this peak")
            continue

        # 使用较密集的采样率
        dt_device = np.mean(np.diff(t_device))
        dt_insta = np.mean(np.diff(t_insta))
        dt_common = min(dt_device, dt_insta)
        num_samples = int((t_common_end - t_common_start) / dt_common) + 1
        t_common = np.linspace(t_common_start, t_common_end, num_samples)

        print(
            f"  [Range] Common time window: [{t_common_start:.3f}, {t_common_end:.3f}] ({num_samples} samples)"
        )

        # 精细对齐
        fine_delay, error = optimize_time_delay(
            t_device,
            t_insta,
            gyro_mag_device,
            gyro_mag_insta,
            t_common,
            (t_insta[0], t_insta[-1]),
            initial_guess=coarse_delay_candidate,
            verbose=True,
        )

        results.append(
            {
                "peak_idx": peak_idx,
                "coarse_delay": coarse_delay_candidate,
                "fine_delay": fine_delay,
                "error": error,
                "peak_corr": peak_corr,
                "t_common": t_common,
            }
        )

    # ===== 第3步: 选择最好的结果（误差最小）=====
    if not results:
        print("[ERROR] No valid result was found, aborting")
        return

    best_result = min(results, key=lambda x: x["error"])
    best_idx = best_result["peak_idx"]

    print("\n" + "=" * 60)
    print("[INFO] Optimization results for the three candidate peaks")
    print("=" * 60)
    for r in results:
        marker = " [best]" if r["peak_idx"] == best_idx else ""
        print(
            f"  Peak {r['peak_idx']+1}: coarse delay {r['coarse_delay']*1000:8.3f}ms, "
            f"fine delay {r['fine_delay']*1000:8.3f}ms, "
            f"error {r['error']:.6f}{marker}"
        )

    time_delay = best_result["fine_delay"]
    t_common = best_result["t_common"]

    # 绘制精细对齐后的IMU波形对比
    plot_fine_alignment_gyro_comparison(
        t_device,
        t_insta,
        gyro_mag_device,
        gyro_mag_insta,
        time_delay,
        best_result["coarse_delay"],
    )

    # ===== 打印时间同步结果 =====
    print_time_sync_results(time_delay)

    print("final time delay (s):", time_delay - data["offset"])


if __name__ == "__main__":
    main()
