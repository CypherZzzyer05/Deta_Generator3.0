# 服务器安装与搬移

Python3.10或更新、Windows、Lumerical FDTD和可用许可。新模块不会替换deta_generator2。

有网络时安装：

```powershell
python -m pip install .\artifacts\deta_generator3-3.0.0-py3-none-any.whl
```

无网络服务器：在与服务器Python版本和Windows架构相同的联网环境准备依赖：

```powershell
python -m pip download --dest wheels .\artifacts\deta_generator3-3.0.0-py3-none-any.whl
```

搬移wheels目录及本项目后：

```powershell
python -m pip install --no-index --find-links wheels deta-generator3==3.0.0
$env:LUMERICAL_LUMAPI_PATH = 'E:\Lumerical\ANSYS Inc\v251\Lumerical\api\python\lumapi.py'
python -m deta_generator3 --version
```

任务的job.json、materials和shard/merged目录可以整体搬移。相对材料路径随任务走，求解器安装路径通过环境变量或`--lumapi`指定。不要复制Lumerical安装文件、许可证日志或旧版生成数据到新任务目录。

线程、进程数和输出路径不改变物理身份；数值网格、材料表、波段、观察面、数值代码和几何计划必须一致。实际原生版本或材料拟合不同时不能继续混算同一数据身份。任务中断后先用status核对旧工作进程，确认其终态再续跑。

`merged`包含自己的样本文件和training_manifest，可独立交付；同磁盘硬链接在普通复制/压缩后成为正常独立文件。打包merged时不要再重复打包所有shard样本。日志和失败工程可按需另附，不能将它们当成成功训练数据。
