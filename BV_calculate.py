import math
from errno import EMFILE


def calculate_breakdown_voltage(N_d, epsilon_r, E_critical):
    """
    计算p-n结的击穿电压。

    参数:
    N_d: 施主掺杂浓度（单位: cm^-3）
    N_a: 受主掺杂浓度（单位: cm^-3）
    epsilon_r: 材料的相对介电常数
    E_critical: 临界电场（单位: V/cm）

    返回:
    击穿电压（单位: V）
    """
    # 真空介电常数 (F/cm)
    epsilon_0 = 8.854187817e-14

    # 计算击穿电压

    V_B = (epsilon_r * epsilon_0 * E_critical * E_critical) / (2 * q * N_0)

    return V_B

# 示例参数（硅材料）
N_d = 2e17  # 施主掺杂浓度 (cm^-3)
N_a = 1e19
N_0 = N_a * N_d /(N_a + N_d)
q = 1.6e-19
epsilon_r = 11.7  # 硅的相对介电常数
E_critical = 4e5 / (1 - (1/3 * math.log10(N_d / 1e16)))
#E_critical = 3e5  # 硅的临界电场 (V/cm)

breakdown_voltage = calculate_breakdown_voltage(N_0, epsilon_r, E_critical)
print(f"击穿电压: {breakdown_voltage:.2f} V")