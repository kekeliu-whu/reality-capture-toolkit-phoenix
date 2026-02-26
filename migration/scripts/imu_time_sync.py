"""
IMU 时间同步工具

该模块用于：
1. 读取两个IMU传感器的数据
2. 通过陀螺仪幅度的B-spline插值进行时间同步
3. 生成可视化分析结果
"""

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy import signal
import matplotlib
from proto.sensors_pb2 import ImuMsgList


# ============================================================================
# 配置参数（需要在matplotlib后端设置前定义）
# ============================================================================

# 是否显示图形
SHOW_PLOTS = False

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

    offset = t2[0] - t1[0]
    print(f"  [时间偏移] IMU2相对于IMU1的初始时间偏移: {offset:.3f} s, t1[0]: {t1[0]:.3f} s, t2[0]: {t2[0]:.3f} s")
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
        "offset": offset,
    }


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
    t1: float,
    t2: float,
    gyro_mag1: np.ndarray,
    gyro_mag2: np.ndarray,
    t_common: np.ndarray,
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

    spl1 = create_bspline_models(t1, gyro_mag1)
    spl2 = create_bspline_models(t2, gyro_mag2)

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


# ============================================================================
# 可视化模块
# ============================================================================


def plot_fine_alignment_gyro_comparison(
    t1: np.ndarray,
    t2: np.ndarray,
    gyro_mag1: np.ndarray,
    gyro_mag2: np.ndarray,
    time_delay: float,
    coarse_delay: float,
) -> None:
    """绘制精细对齐后的IMU原始数据对比图"""
    print("[图表] 正在绘制精细对齐后的IMU波形对比...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制IMU1的原始数据
    ax.plot(
        t1,
        gyro_mag1,
        "g-",
        linewidth=1.5,
        label="IMU1 陀螺仪幅度（原始）",
        alpha=0.8,
    )

    # 绘制IMU2精细对齐后的数据（时间轴加上总延迟）
    t2_aligned = t2 + time_delay
    ax.plot(
        t2_aligned,
        gyro_mag2,
        "r-",
        linewidth=1.5,
        label=f"IMU2 陀螺仪幅度（精细对齐后，总延迟: {time_delay*1000:.3f}ms）",
        alpha=0.8,
    )

    ax.set_xlabel("时间 (s)", fontsize=12)
    ax.set_ylabel("陀螺仪幅度 (rad/s)", fontsize=12)
    ax.set_title(
        f"精细对齐后的IMU波形对比（原始数据）",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fine_alignment_gyro_comparison.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()

    print("  [保存] fine_alignment_gyro_comparison.png")


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

    print(
        f"[时间轴] 公共时间范围: [{t_common_start:.3f}, {t_common_end:.3f}] ({num_samples} 个采样点)"
    )

    # ===== 第4步: 精细对齐（基于B-spline优化）=====
    time_delay = optimize_time_delay(
        t1,
        t2,
        gyro_mag1,
        gyro_mag2,
        t_common,
        (t2[0], t2[-1]),
        initial_guess=coarse_delay,
    )

    # 绘制精细对齐后的IMU波形对比
    plot_fine_alignment_gyro_comparison(
        t1, t2, gyro_mag1, gyro_mag2, time_delay, coarse_delay
    )

    # ===== 打印时间同步结果 =====
    print_time_sync_results(time_delay)

    print("final time delay (s):", time_delay - data["offset"])


if __name__ == "__main__":
    main()
