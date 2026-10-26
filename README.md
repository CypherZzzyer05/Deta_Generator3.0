# Deta Generator 3.0

参照本机 Deta Generator 2.0 的独立安装、CPU 分片并行、断点续跑和状态监控方式，为光谱＋场强多模态逆向设计生成配对数据。包名 `deta-generator3`，模块名 `deta_generator3`；与 2.0 共存。无需 CUDA、PyTorch 或原 LumOpt_test 项目。

## 固定物理约定

|项目|定义|
|---|---|
|自上而下结构|空气 / As₂Se₃ 四聚体 / Au100 nm / **有限玻璃350 nm** / 空气|
|z 坐标|四聚体0…h；金−100…0；玻璃−450…−100 nm|
|七参数顺序|`p, dx, dy, h, l, alpha_1and3, alpha_2and4`；前五项nm，后两项deg|
|dx、dy|相邻单体**中心距**，四个中心为 `(±dx/2, ±dy/2)`|
|角度|从+z看逆时针；I/III共用奇角，II/IV共用偶角；方形角模90°，范围[-45,45)|
|参数范围|p700–900、dx/dy300–400、h200–350、l160–220 nm；工程探索范围|
|激励／边界|正入射沿−z，x线偏振；XY周期、Z PML|
|光谱|600–1400nm，等波长1000点；保存带符号R、T和1−R−T|
|场观察|z100/400nm，两平面×600:100:1400nm九波长|
|模型输入|固定1600nm视场，128×128，轴[平面,波长,y,x]，x/y递增|
|材料|Joseph 400nm薄膜实测As₂Se₃ n,k；Au与SiO₂为安装版Palik|
|当前数值协议|结构10nm、金层z5nm、功率采样降采样2；原16ps上限、自动关断1e-5|

与2.0默认的半无限/PML玻璃、边缘间隙参数和700–1700nm光谱不同，**不能在旧任务目录续算或混合身份**。原生复场先做零Bloch周期复数插值，再计算绝对强度；不对每图除最大值。数据始终标为 `pipeline_validation`，不宣称完成全域网格或材料精度认证。

## 安装与单样本试算

Windows服务器需安装可用的Lumerical FDTD和许可，Python≥3.10。依赖仅NumPy、SciPy、psutil。

```powershell
python -m pip install .
$env:LUMERICAL_LUMAPI_PATH = 'D:\Program Files\Lumerical\v241\api\python\lumapi.py'
python -m deta_generator3 --version
python -m deta_generator3 init --job-dir jobs/pilot --samples 3 --threads 4
python -m deta_generator3 preflight --config jobs/pilot/job.json
python -m deta_generator3 generate --config jobs/pilot/job.json --limit 1
```

可安装 `artifacts/deta_generator3-3.0.0-py3-none-any.whl`。`lumapi`使用安装版自带API，不从PyPI另装。服务器v251也可通过同一环境变量指定对应 `lumapi.py`；多个版本共存时需要明确选择。`doctor`和`preflight`只建模／读网格／检查材料和资源，不推进时间求解。

## 像2.0一样同时计算多个样本

```powershell
python -m deta_generator3 parallel --job-dir jobs/paired1000 --samples 1000 --jobs 10
python -m deta_generator3 parallel --job-dir jobs/paired1000 --samples 1000 --jobs 10 --parallel-action status
```

`--jobs 10`代表10个独立几何同时求解；每个样本的线程默认按可用CPU预算除以并发数分配。任务先固定全部几何和划分，再建立多个小分片，完成一个分片后补上下一个。原生预检查先于并行求解。需要足够内存和许可证并发容量；首次运行先用单样本预检查获得每例内存估计，再选择服务器并发数。

中断后用相同命令续跑。样本数和分片布局固定，改变时使用新目录；线程和安装路径可随机器调整，物理／材料／观察算子改变会被拒绝。控制器中断不杀掉已运行的工作进程，重启时检查旧进程身份。`.run/`提供PyCharm共享配置。默认失败案例保留并跳过，继续其他几何；仅串行明确指定 `--retry-failed` 才重试，旧错误及失败工程保留，新尝试写入独立retry子目录。

### 引擎消失但工作进程仍存活时

单独的低引擎数量提示不能证明故障。如果所有运行中样本的日志已至少一小时未更新、进度未达100%，并确认无原生求解引擎，可按保留现场的中断恢复处理：先在原控制器终端按Ctrl+C，随后运行独立辅助脚本。

```powershell
python scripts/recover_stalled_task.py --job-dir jobs/paired1000
```

脚本要求原控制器已释放锁，并再次检查引擎、日志和进程创建身份。先将任务元数据、全部分片清单、工作进程日志及当前中断样本的工程／日志复制到任务目录下的`recovery_*`并核对SHA，再只终止当前任务已确认的工作进程及其FDTD CAD子进程。MPI服务、其他项目进程、成功NPZ和失败记录保留；不删除锁文件、不修改清单、不自动重启。发现引擎仍在运行、观察权限不足、进度变化、PID复用或未提交数值样本时停止恢复，需检查具体提示。该工具的进程／文件夹测试使用模拟对象，不是真实服务器恢复证明。

脚本输出`preserved_and_released_no_restart`后，可先以两个并发恢复原任务：

```powershell
python -m deta_generator3 parallel --job-dir jobs/paired1000 --samples 1000 --jobs 2 --threads 4 --parallel-action run
```

这会跳过已成功及已失败的样本，继续中断／pending样本，保留原样本数、分片和划分。先确认新日志持续更新及实际引擎运行，再考虑增加并发。恢复操作不等于根因已经查明，仍需结合许可证、内存、磁盘和原生软件诊断。

## 输出与磁盘

- `job.json`、`materials/`：可搬移配置和原材料表。
- `shard*/manifest.json`：完整原计划、固定划分、状态、样本SHA、原生版本和实际材料拟合。
- `shard*/samples/sample_XXXXXX.npz`：R/T/A、七参数、场强、热点掩膜／标签、物理坐标、原生复数E/H和近场Poynting。
- `merged/training_manifest.json`：可由本项目 `ShardedPairedDataset` 读取的配对清单；成功状态转换为`complete`，失败仍保留。
- `merged/dataset.npz`：光谱、标签和场样本路径索引；**不再堆叠复制全部场图**。
- `parallel_logs/`、`parallel_workers.json`：工作进程日志和身份；状态监控区分Python工作进程和真正观察到的FDTD引擎。

同磁盘合并使用硬链接共享不可变的样本文件；跨磁盘才校验后复制。单样本只保留一套原生E/H，省去重复的128×128复场缓存；固定视场强度和能流仍保存。默认成功后仅删除该样本自己的FSP临时工程，保留配对NPZ、原始复场和日志；`init --keep-projects`可保留全部成功工程。失败FSP和拒绝谱值保留，不当作训练样本。

给本项目传回数据时，可打包**完整merged目录**并附带job.json和materials，避免同时把全部shard样本再打包一份；失败诊断可另包。训练时读取 `training_manifest.json`，仅将train划分用于优化；直接装载完整训练接口需要train/val/test各有成功样本，见summary.json的dataset_ready。新任务默认seed20261023，原训练／验证／测试几何及镜像仍需在跨数据集接入时核对。

## 质量门槛与高强度区域

必须status2关断、1000点有效R/T、功率范围／和在0.03容差内；不裁剪负谱值。As₂Se₃全波段拟合最大复折射率误差≤0.03、无显著增益。原生E/H必须同坐标，固定窗口重建检查周期接缝。实际拟合、软件版本、数值设置或标签冲突时拒绝合并。

强度为 `(|Ex|²+|Ey|²+|Ez|²)/|E_inc|²`，入射振幅1 V/m。热点用种子阈值max(2,0.5×峰值)、生长阈值max(2,0.2×峰值)，固定窗口边缘不拼接。保存热点面积、亮度占比、连通区域及形态主轴。`0.5 Re(E×H*)`为近场功率流，不等同于远场逸散角；场强亮度也不是色散材料储能密度。生成器不输出虚构的远场角度标签。

开发检查：`python -m unittest discover -s tests -v`。测试用求解器仅验证调度和数据完整性；原生预检查、真实求解结果及其限制另见 `docs/validation.md`。来源和材料说明见 `SOURCE.md` 与包内材料README。
