# 中文使用指南

本项目实现修订论文中的中央 PPO 控制器与五个固定专家：PEA、TCA、VGA、CVfA 和 GIA。控制器训练包含真正的 LoRA/决策头梯度更新，专家在控制器训练期间保持冻结。

## 准备环境和模型

使用 Python 3.10 及以上版本，先安装与你的硬件匹配的 PyTorch，然后在项目根目录执行：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[train,retrieval,dev]"
rl-mac-mkg --help
```

模型目录已经建立，角色说明见 [models/README.md](../models/README.md)。把 Qwen3-8B、Qwen3-VL-8B-Instruct、CLIP、SentenceTransformer 的标准文件放到对应目录；专家 LoRA 适配器可以自行提供，或使用 `train-specialist` 训练。代码不会自动下载权重。

`configs/experiment.yaml` 中的路径相对于配置文件解析。各角色可分别设置 CPU/GPU 和精度。完整 8B 训练所需内存取决于精度、序列长度和同时驻留模型数；CPU 小模型测试用于检查算法和软件链路。

## 准备数据

按照 [data-format.md](data-format.md) 建立实体、来源、模式和标注文件。每个来源需要真实 URL/标识、获取时间、许可和内容 SHA-256。文本跨度使用原始文档字符位置；图像需要实际文件、图像 ID 与安全审查状态。

执行 `split` 建立实体与来源隔离的划分，然后将输出清单写入配置。`validate` 检查文件哈希、来源分配和跨划分泄漏。训练过程中只暴露当前分区的实体；通用本体节点需要通过 `shared_entity_ids` 显式声明，且不能属于任何种子实体分区。

## 训练顺序

1. 在 specialist 分区训练文本和视觉适配器，在 verifier 分区训练支持判断适配器。
2. 在 calibration 分区用 `score-verifier` 收集事实/视觉支持 logits，再用 `calibrate` 拟合两个温度。
3. 从 validation 调用测量拟合动作成本，在验证集选择未报告的奖励系数并固定。
4. 使用带操作与目标标签的 PPO 训练分区轨迹执行 `train-controller --stage bc`。
5. 从 BC 检查点执行 `train-controller --stage ppo --from-bc ...`。
6. 需要继续中断的运行时使用 `--resume` 并选择新的输出目录。

完整命令见 [training.md](training.md) 和主 README。论文已给出的 PPO、预算、跳数和代理分数参数已写入配置；未报告项的具体实现和初始值在 [method-mapping.md](method-mapping.md) 中明确区分。

## 推理、评估与上传

测试时把数据分区改为 `test`，使用验证选定的控制器检查点，运行 `infer`。输出包含图谱、来源锚点、候选状态、冲突、证据账本和完整轨迹。弃权候选不会写成断言。

`evaluate` 计算关系/图像属性 P/R/F1；`evaluate-retrieval` 计算固定检索结果的 Recall@K/MRR；`statistics` 完成配对 bootstrap、置换检验与 Holm 校正。

上传代码前执行测试、Ruff 与打包命令。模型文件、运行输出、原始数据、缓存和密钥默认不进入 Git。代码已经与原 `MAC-KG-main` 独立，原工程可以继续作为旧方法参考。
