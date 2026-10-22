# As₂Se₃：600–1400 nm 的复折射率

`as2se3_joseph_400nm.csv` 为 `wavelength_nm,n,k`，677 行，245.8–1688 nm。转换只把波长从 μm 乘以 1000，没有平滑、补值或将 k 置零。

原始数据保存在 `as2se3_joseph_400nm_source.yml`，来自 [refractiveindex.info 数据库的原始记录](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/As2Se3/nk/Joseph-400nm.yml)。原文件声明 CC0 1.0；数值由论文作者 Joby Joseph 提供，记录对应**玻璃上的 400 nm 薄膜**。关联论文：[Joseph, Sarkar & Joseph, ACS Applied Materials & Interfaces 12, 46519–46529 (2020)](https://doi.org/10.1021/acsami.0c12525)。以源文件中的论文年份为准，数据库页面的展示标签另写 2021。

2026-10-19 下载的原 YAML SHA256：`3c7967a9d7fccab62f978d31420d4382861f1f6604e260f2eecf10b9d6e7a13b`。

| 波长 nm | n（线性插值） | k（线性插值） |
| --- | --- | --- |
| 600 | 3.140295 | 0.161469 |
| 700 | 2.900506 | 0.023627 |
| 1000 | 2.695615 | 0.002999 |
| 1400 | 2.648344 | 0.010266 |

短波端的吸收不可忽略；600 nm 的强度吸收长度 `λ/(4πk)` 约为 296 nm。600–1400 nm 是本次用户指定的研究窗口，尚不能认定它优于中红外窗口。

用 400 nm 连续薄膜的光学常数近似 200–350 nm 高的图案化单体，仍需通过实际制备材料、厚度和退火条件核对。当前数据质量标记保持 `pipeline_validation`；正式研究应替换为匹配工艺的实测 n,k。

FDTD 按 `[frequency_Hz, (n+ik)^2]` 导入，并比较拟合 n,k 与原始表插值值；最大复数折射率绝对误差超过 0.03 或出现显著增益时拒绝求解。每个样本保存拟合报告，采样配置指纹包含 CSV 内容 SHA256。[Ansys 导入格式](https://optics.ansys.com/hc/en-us/articles/360034409654-setmaterial-Script-command)与[材料拟合说明](https://optics.ansys.com/hc/en-us/articles/360034915093-Creating-new-sampled-data-materials-in-FDTD)。

现已将拟合检查修正为完整600–1400nm、1000个波长点。旧代码每隔10点取样，末点1392.8nm也被误用作拟合边界；[getfdtdindex](https://optics.ansys.com/hc/en-us/articles/360034409694-getfdtdindex-Script-command) 的拟合边界应与实际仿真源一致。只读复核已保存的粗网格sample39与细网格sample6工程，完整波段最大复折射率误差均为 **0.01223143**，低于0.03，工程哈希均未改变。该复核检查相同的材料模型，不等同于网格收敛或器件实测准确性。
