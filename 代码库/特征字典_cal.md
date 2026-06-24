# 传感器特征字典（D4交付物）

## 说明
- 增加波形因子、Jerk、谱平坦度、排列熵、微振动能量、重力分布、航向统计等特征
- 专门针对静坐/站立混淆问题优化
- 新增重力X/Z比值、俯仰×横滚交互特征
- 相对姿态角消除佩戴误差

## 特征列表

### 时域基础（36个特征）
| 特征ID | 特征名称 | 定义 | 适用传感器 | 物理含义 |
|--------|----------|------|------------|----------|
| time_basic_acc_x_kurtosis | acc_x峰度 | 衡量acc_x信号分布的陡峭程度 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_max | acc_x最大值 | 窗口内acc_x信号的最大值 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_mean | acc_x时域均值 | 窗口内acc_x信号的平均值 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_median | acc_x中位数 | 窗口内acc_x信号的中位数 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_min | acc_x最小值 | 窗口内acc_x信号的最小值 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_peak2peak | acc_x峰峰值 | 窗口内acc_x信号的最大值与最小值之差 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_rms | acc_xRMS | 窗口内acc_x信号的均方根 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_skewness | acc_x偏度 | 衡量acc_x信号分布的对称性 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_sma | acc_xSMA | 窗口内acc_x信号绝对值的均值 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_std | acc_x时域标准差 | 窗口内acc_x信号的标准差 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_var | acc_x时域方差 | 窗口内acc_x信号的方差 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_x_zerocross_rate | acc_x过零率 | 单位时间内信号穿过零点的次数 | acc_x | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_kurtosis | acc_y峰度 | 衡量acc_y信号分布的陡峭程度 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_max | acc_y最大值 | 窗口内acc_y信号的最大值 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_mean | acc_y时域均值 | 窗口内acc_y信号的平均值 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_median | acc_y中位数 | 窗口内acc_y信号的中位数 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_min | acc_y最小值 | 窗口内acc_y信号的最小值 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_peak2peak | acc_y峰峰值 | 窗口内acc_y信号的最大值与最小值之差 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_rms | acc_yRMS | 窗口内acc_y信号的均方根 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_skewness | acc_y偏度 | 衡量acc_y信号分布的对称性 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_sma | acc_ySMA | 窗口内acc_y信号绝对值的均值 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_std | acc_y时域标准差 | 窗口内acc_y信号的标准差 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_var | acc_y时域方差 | 窗口内acc_y信号的方差 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_y_zerocross_rate | acc_y过零率 | 单位时间内信号穿过零点的次数 | acc_y | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_kurtosis | acc_z峰度 | 衡量acc_z信号分布的陡峭程度 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_max | acc_z最大值 | 窗口内acc_z信号的最大值 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_mean | acc_z时域均值 | 窗口内acc_z信号的平均值 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_median | acc_z中位数 | 窗口内acc_z信号的中位数 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_min | acc_z最小值 | 窗口内acc_z信号的最小值 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_peak2peak | acc_z峰峰值 | 窗口内acc_z信号的最大值与最小值之差 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_rms | acc_zRMS | 窗口内acc_z信号的均方根 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_skewness | acc_z偏度 | 衡量acc_z信号分布的对称性 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_sma | acc_zSMA | 窗口内acc_z信号绝对值的均值 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_std | acc_z时域标准差 | 窗口内acc_z信号的标准差 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_var | acc_z时域方差 | 窗口内acc_z信号的方差 | acc_z | 反映信号的幅值、波动和分布特性 |
| time_basic_acc_z_zerocross_rate | acc_z过零率 | 单位时间内信号穿过零点的次数 | acc_z | 反映信号的幅值、波动和分布特性 |

### 时域高阶（21个特征）
| 特征ID | 特征名称 | 定义 | 适用传感器 | 物理含义 |
|--------|----------|------|------------|----------|
| time_adv_acc_x_autocorr_lag1 | acc_x滞后1自相关 | 窗口内acc_x信号相邻采样点的相关性 | acc_x | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_x_clearance_factor | acc_x裕度因子 | 峰值/方根幅值平方，反映冲击与整体能量的关系 | acc_x | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_x_crest_factor | acc_x峰值因子 | 峰值/RMS，反映冲击强度 | acc_x | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_x_impulse_factor | acc_x脉冲因子 | 峰值/绝对均值，对冲击敏感 | acc_x | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_x_iqr | acc_x四分位距 | acc_x信号75%与25%分位数之差 | acc_x | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_x_mean_abs_dev | acc_x平均绝对偏差 | 窗口内acc_x信号与均值差的绝对值的平均 | acc_x | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_x_shape_factor | acc_x波形因子 | RMS/绝对均值，反映波形尖锐程度 | acc_x | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_y_autocorr_lag1 | acc_y滞后1自相关 | 窗口内acc_y信号相邻采样点的相关性 | acc_y | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_y_clearance_factor | acc_y裕度因子 | 峰值/方根幅值平方，反映冲击与整体能量的关系 | acc_y | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_y_crest_factor | acc_y峰值因子 | 峰值/RMS，反映冲击强度 | acc_y | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_y_impulse_factor | acc_y脉冲因子 | 峰值/绝对均值，对冲击敏感 | acc_y | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_y_iqr | acc_y四分位距 | acc_y信号75%与25%分位数之差 | acc_y | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_y_mean_abs_dev | acc_y平均绝对偏差 | 窗口内acc_y信号与均值差的绝对值的平均 | acc_y | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_y_shape_factor | acc_y波形因子 | RMS/绝对均值，反映波形尖锐程度 | acc_y | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_z_autocorr_lag1 | acc_z滞后1自相关 | 窗口内acc_z信号相邻采样点的相关性 | acc_z | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_z_clearance_factor | acc_z裕度因子 | 峰值/方根幅值平方，反映冲击与整体能量的关系 | acc_z | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_z_crest_factor | acc_z峰值因子 | 峰值/RMS，反映冲击强度 | acc_z | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_z_impulse_factor | acc_z脉冲因子 | 峰值/绝对均值，对冲击敏感 | acc_z | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_z_iqr | acc_z四分位距 | acc_z信号75%与25%分位数之差 | acc_z | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_z_mean_abs_dev | acc_z平均绝对偏差 | 窗口内acc_z信号与均值差的绝对值的平均 | acc_z | 反映信号的时序相关性和抗噪特性 |
| time_adv_acc_z_shape_factor | acc_z波形因子 | RMS/绝对均值，反映波形尖锐程度 | acc_z | 反映信号的时序相关性和抗噪特性 |

### 频域（99个特征）
| 特征ID | 特征名称 | 定义 | 适用传感器 | 物理含义 |
|--------|----------|------|------------|----------|
| freq_acc_x_3db_bandwidth | acc_x3dB带宽 | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_band1_0.5-2Hz_energy_ratio | ... | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_band2_2-5Hz_energy_ratio | ... | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_band3_5-10Hz_energy_ratio | ... | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_band4_10-25Hz_energy_ratio | ... | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_main_freq | acc_x主频 | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_rms_freq | acc_x均方根频率 | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_spectral_centroid | acc_x频谱质心 | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_spectral_entropy | acc_x谱熵 | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_spectral_flatness | acc_x谱平坦度 | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_x_spectral_slope | acc_x谱斜率 | ... | acc_x | 反映信号的频率成分和能量分布 |
| freq_acc_y_3db_bandwidth | acc_y3dB带宽 | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_band1_0.5-2Hz_energy_ratio | ... | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_band2_2-5Hz_energy_ratio | ... | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_band3_5-10Hz_energy_ratio | ... | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_band4_10-25Hz_energy_ratio | ... | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_main_freq | acc_y主频 | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_rms_freq | acc_y均方根频率 | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_spectral_centroid | acc_y频谱质心 | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_spectral_entropy | acc_y谱熵 | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_spectral_flatness | acc_y谱平坦度 | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_y_spectral_slope | acc_y谱斜率 | ... | acc_y | 反映信号的频率成分和能量分布 |
| freq_acc_z_3db_bandwidth | acc_z3dB带宽 | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_band1_0.5-2Hz_energy_ratio | ... | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_band2_2-5Hz_energy_ratio | ... | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_band3_5-10Hz_energy_ratio | ... | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_band4_10-25Hz_energy_ratio | ... | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_main_freq | acc_z主频 | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_rms_freq | acc_z均方根频率 | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_spectral_centroid | acc_z频谱质心 | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_spectral_entropy | acc_z谱熵 | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_spectral_flatness | acc_z谱平坦度 | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_acc_z_spectral_slope | acc_z谱斜率 | ... | acc_z | 反映信号的频率成分和能量分布 |
| freq_gyro_x_3db_bandwidth | gyro_x3dB带宽 | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_band1_0.5-2Hz_energy_ratio | ... | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_band2_2-5Hz_energy_ratio | ... | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_band3_5-10Hz_energy_ratio | ... | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_band4_10-25Hz_energy_ratio | ... | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_main_freq | gyro_x主频 | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_rms_freq | gyro_x均方根频率 | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_spectral_centroid | gyro_x频谱质心 | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_spectral_entropy | gyro_x谱熵 | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_spectral_flatness | gyro_x谱平坦度 | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_x_spectral_slope | gyro_x谱斜率 | ... | gyro_x | 反映信号的频率成分和能量分布 |
| freq_gyro_y_3db_bandwidth | gyro_y3dB带宽 | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_band1_0.5-2Hz_energy_ratio | ... | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_band2_2-5Hz_energy_ratio | ... | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_band3_5-10Hz_energy_ratio | ... | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_band4_10-25Hz_energy_ratio | ... | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_main_freq | gyro_y主频 | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_rms_freq | gyro_y均方根频率 | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_spectral_centroid | gyro_y频谱质心 | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_spectral_entropy | gyro_y谱熵 | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_spectral_flatness | gyro_y谱平坦度 | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_y_spectral_slope | gyro_y谱斜率 | ... | gyro_y | 反映信号的频率成分和能量分布 |
| freq_gyro_z_3db_bandwidth | gyro_z3dB带宽 | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_band1_0.5-2Hz_energy_ratio | ... | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_band2_2-5Hz_energy_ratio | ... | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_band3_5-10Hz_energy_ratio | ... | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_band4_10-25Hz_energy_ratio | ... | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_main_freq | gyro_z主频 | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_rms_freq | gyro_z均方根频率 | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_spectral_centroid | gyro_z频谱质心 | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_spectral_entropy | gyro_z谱熵 | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_spectral_flatness | gyro_z谱平坦度 | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_gyro_z_spectral_slope | gyro_z谱斜率 | ... | gyro_z | 反映信号的频率成分和能量分布 |
| freq_mag_x_3db_bandwidth | mag_x3dB带宽 | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_band1_0.5-2Hz_energy_ratio | ... | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_band2_2-5Hz_energy_ratio | ... | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_band3_5-10Hz_energy_ratio | ... | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_band4_10-25Hz_energy_ratio | ... | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_main_freq | mag_x主频 | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_rms_freq | mag_x均方根频率 | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_spectral_centroid | mag_x频谱质心 | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_spectral_entropy | mag_x谱熵 | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_spectral_flatness | mag_x谱平坦度 | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_x_spectral_slope | mag_x谱斜率 | ... | mag_x | 反映信号的频率成分和能量分布 |
| freq_mag_y_3db_bandwidth | mag_y3dB带宽 | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_band1_0.5-2Hz_energy_ratio | ... | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_band2_2-5Hz_energy_ratio | ... | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_band3_5-10Hz_energy_ratio | ... | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_band4_10-25Hz_energy_ratio | ... | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_main_freq | mag_y主频 | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_rms_freq | mag_y均方根频率 | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_spectral_centroid | mag_y频谱质心 | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_spectral_entropy | mag_y谱熵 | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_spectral_flatness | mag_y谱平坦度 | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_y_spectral_slope | mag_y谱斜率 | ... | mag_y | 反映信号的频率成分和能量分布 |
| freq_mag_z_3db_bandwidth | mag_z3dB带宽 | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_band1_0.5-2Hz_energy_ratio | ... | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_band2_2-5Hz_energy_ratio | ... | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_band3_5-10Hz_energy_ratio | ... | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_band4_10-25Hz_energy_ratio | ... | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_main_freq | mag_z主频 | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_rms_freq | mag_z均方根频率 | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_spectral_centroid | mag_z频谱质心 | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_spectral_entropy | mag_z谱熵 | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_spectral_flatness | mag_z谱平坦度 | ... | mag_z | 反映信号的频率成分和能量分布 |
| freq_mag_z_spectral_slope | mag_z谱斜率 | ... | mag_z | 反映信号的频率成分和能量分布 |

### 时频（18个特征）
| 特征ID | 特征名称 | 定义 | 适用传感器 | 物理含义 |
|--------|----------|------|------------|----------|
| timefreq_acc_x_cwt_energy_mean | acc_xCWT能量均值 | ... | acc_x | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_x_cwt_entropy | acc_xCWT熵 | ... | acc_x | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_x_main_scale_energy_ratio | acc_x主尺度能量占比 | ... | acc_x | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_y_cwt_energy_mean | acc_yCWT能量均值 | ... | acc_y | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_y_cwt_entropy | acc_yCWT熵 | ... | acc_y | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_y_main_scale_energy_ratio | acc_y主尺度能量占比 | ... | acc_y | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_z_cwt_energy_mean | acc_zCWT能量均值 | ... | acc_z | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_z_cwt_entropy | acc_zCWT熵 | ... | acc_z | 反映非平稳信号的时变频谱特性 |
| timefreq_acc_z_main_scale_energy_ratio | acc_z主尺度能量占比 | ... | acc_z | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_x_cwt_energy_mean | gyro_xCWT能量均值 | ... | gyro_x | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_x_cwt_entropy | gyro_xCWT熵 | ... | gyro_x | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_x_main_scale_energy_ratio | gyro_x主尺度能量占比 | ... | gyro_x | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_y_cwt_energy_mean | gyro_yCWT能量均值 | ... | gyro_y | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_y_cwt_entropy | gyro_yCWT熵 | ... | gyro_y | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_y_main_scale_energy_ratio | gyro_y主尺度能量占比 | ... | gyro_y | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_z_cwt_energy_mean | gyro_zCWT能量均值 | ... | gyro_z | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_z_cwt_entropy | gyro_zCWT熵 | ... | gyro_z | 反映非平稳信号的时变频谱特性 |
| timefreq_gyro_z_main_scale_energy_ratio | gyro_z主尺度能量占比 | ... | gyro_z | 反映非平稳信号的时变频谱特性 |

### 融合特征（38个特征）
| 特征ID | 特征名称 | 定义 | 适用传感器 | 物理含义 |
|--------|----------|------|------------|----------|
| fusion_acc_corr_ax_ay | 加速度ax-ay互相关 | ... | 三轴加速度 | 反映多传感器间的物理耦合关系 |
| fusion_acc_corr_ax_az | 加速度ax-az互相关 | ... | 三轴加速度 | 反映多传感器间的物理耦合关系 |
| fusion_acc_corr_ay_az | 加速度ay-az互相关 | ... | 三轴加速度 | 反映多传感器间的物理耦合关系 |
| fusion_acc_mag_mean | 合加速度均值 | ... | 三轴加速度 | 反映多传感器间的物理耦合关系 |
| fusion_acc_mag_permutation_entropy | 合加速度排列熵 | 时间序列符号化后的熵，越小越规律 | 加速度 | 反映多传感器间的物理耦合关系 |
| fusion_acc_mag_std | 合加速度标准差 | ... | 三轴加速度 | 反映多传感器间的物理耦合关系 |
| fusion_accax_gyrogx_corr | 加速度ax-陀螺仪gx互相关 | ... | ax+gx | 反映多传感器间的物理耦合关系 |
| fusion_accax_gyrogy_corr | 加速度ax-陀螺仪gy互相关 | ... | ax+gy | 反映多传感器间的物理耦合关系 |
| fusion_accax_gyrogz_corr | 加速度ax-陀螺仪gz互相关 | ... | ax+gz | 反映多传感器间的物理耦合关系 |
| fusion_accay_gyrogx_corr | 加速度ay-陀螺仪gx互相关 | ... | ay+gx | 反映多传感器间的物理耦合关系 |
| fusion_accay_gyrogy_corr | 加速度ay-陀螺仪gy互相关 | ... | ay+gy | 反映多传感器间的物理耦合关系 |
| fusion_accay_gyrogz_corr | 加速度ay-陀螺仪gz互相关 | ... | ay+gz | 反映多传感器间的物理耦合关系 |
| fusion_accaz_gyrogx_corr | 加速度az-陀螺仪gx互相关 | ... | az+gx | 反映多传感器间的物理耦合关系 |
| fusion_accaz_gyrogy_corr | 加速度az-陀螺仪gy互相关 | ... | az+gy | 反映多传感器间的物理耦合关系 |
| fusion_accaz_gyrogz_corr | 加速度az-陀螺仪gz互相关 | ... | az+gz | 反映多传感器间的物理耦合关系 |
| fusion_grav_horiz_angle_mean | 重力水平投影角均值 | 重力在X-Z平面的投影角度均值 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_grav_horiz_angle_std | 重力水平投影角标准差 | 重力在X-Z平面的投影角度波动 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_grav_ratio_x | 重力X轴占比 | 重力在X轴分量的平均比例 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_grav_ratio_xz | 重力 X/Z 比值 | 重力在X轴与Z轴分量的比值，描述传感器绕Y轴的倾斜方向 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_grav_ratio_y | 重力Y轴占比 | 重力在Y轴分量的平均比例 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_grav_ratio_z | 重力Z轴占比 | 重力在Z轴分量的平均比例 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_gyro_mag_mean | 合角速度均值 | ... | 三轴陀螺仪 | 反映多传感器间的物理耦合关系 |
| fusion_gyro_mag_std | 合角速度标准差 | ... | 三轴陀螺仪 | 反映多传感器间的物理耦合关系 |
| fusion_heading_range | 航向角范围 | 窗口内航向角最大变化范围 | 磁力计 | 反映多传感器间的物理耦合关系 |
| fusion_heading_std | 航向角标准差 | 窗口内航向角标准差，反映身体朝向稳定性 | 磁力计 | 反映多传感器间的物理耦合关系 |
| fusion_jerk_mag_mean | 合Jerk均值 | 加速度导数的模长均值 | 加速度导数 | 反映多传感器间的物理耦合关系 |
| fusion_jerk_mag_std | 合Jerk标准差 | 加速度导数的模长标准差 | 加速度导数 | 反映多传感器间的物理耦合关系 |
| fusion_pitch_rel_low_energy | 俯仰角低频能量比 | 0.1-0.5Hz频带能量占比，站立时微小晃动使该值略高 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_pitch_rel_mad | 相对俯仰角绝对偏差 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_pitch_rel_mean | 相对俯仰角均值 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_pitch_rel_range | 相对俯仰角范围 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_pitch_rel_std | 相对俯仰角标准差 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_pitch_roll_interaction | 俯仰角×横滚角交互 | 相对俯仰角与相对横滚角的乘积的均值，反映身体前后倾与侧倾的耦合程度 | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_roll_rel_mad | 相对横滚角绝对偏差 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_roll_rel_mean | 相对横滚角均值 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_roll_rel_range | 相对横滚角范围 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_roll_rel_std | 相对横滚角标准差 | ... | 重力 | 反映多传感器间的物理耦合关系 |
| fusion_vib_energy | 微振动能量 | 0.5-3Hz带通滤波后的加速度能量，站立时比静坐高 | 加速度 | 反映多传感器间的物理耦合关系 |
