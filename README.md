# GB Calculator

一个用于估算模型推理和训练所需显存、显卡数量和训练时长的轻量级仓库。

## 功能

- 支持不同参数规模的模型样本
- 支持多种训练方法：full fine-tuning、LoRA、QLoRA、prompt tuning、inference
- 估算：
  - 权重显存
  - 梯度显存
  - 优化器状态显存
  - 激活显存
  - 训练总显存
  - 所需 GPU 数量
  - 训练时间
- 支持导出 JSON / 表格风格结果

## 快速开始

首次使用建议先安装 Hugging Face 可选依赖，用于自动拉取公开模型配置并写入本地缓存：

```bash
pip install .[hf]
```

查看帮助：

```bash
python -m gb_calculator --help
```

### 第一步：同步模型配置到本地缓存

如果你想用公开模型的全称进行估算（本仓库只推荐使用公开模型全称），可通过下述命令将模型配置同步到本地缓存：

```bash
python -m gb_calculator.sync_models meta-llama/Llama-3.1-8B-Instruct --json
```

同步成功后，模型的 `hidden_size`、`layers`、`seq_len` 等信息会写入 `data/model_cache.json`。

同步输出示例：

```json
{
  "name": "meta-llama/Llama-3.1-8B-Instruct",
  "parameters_billion": 8.0,
  "hidden_size": 4096,
  "layers": 32,
  "seq_len": 4096
}
```

### 第二步：使用模型全称进行估算

```bash
python -m gb_calculator --model meta-llama/Llama-3.1-8B-Instruct --method lora --batch-size 1 --seq-len 4096 --steps 10000 --gpu-vram-gb 24 --precision bf16 --optimizer adamw --gradient-checkpointing
```

CLI 报告输出示例：

```text
【评估对象】meta-llama/Llama-3.1-8B-Instruct，采用 lora 方案。
【显存测算】单卡总显存需求约 17.02 GB，其中权重 15.00 GB、梯度 0.12 GB、优化器状态 0.24 GB、激活 1.66 GB。
【资源建议】在每卡 24.0 GB 的条件下，建议至少准备 1 张 GPU。
【训练时长】按当前 batch size=1、seq_len=4096、steps=10000 估算，完成训练约需 5.18 小时。
【结论】当前配置可以在单卡内运行。
```

如果你希望获得结构化结果，追加 `--json` 即可。

#### 输入参数说明

- `--model`
  - 公开模型的全称，或已经同步到本地缓存的模型名。
  - 示例：`meta-llama/Llama-3.1-8B-Instruct`
  - 说明：支持大小写不敏感，且会自动忽略下划线、连字符和多余空格；例如 `meta-llama/Llama 3.1 8B Instruct` 也可以被识别。
  - 作用：用于查找模型的 `hidden_size`、`layers`、`seq_len` 和参数量等信息。

- `--method`
  - 训练/推理方法。
  - 可选值：`inference`、`full`、`lora`、`qlora`、`prompt_tuning`、`prefix_tuning`
  - 作用：决定权重显存、可训练参数比例、优化器状态开销和吞吐估算方式。

- `--batch-size`
  - 每次前向/反向传播使用的样本数。
  - 默认值：`1`
  - 作用：batch 越大，激活显存通常越高，训练吞吐和总耗时也会变化。

- `--seq-len`
  - 单个样本的输入长度。
  - 默认值：`2048`
  - 作用：主要影响激活显存和训练计算量。

- `--steps`
  - 训练步数。
  - 默认值：`10000`
  - 作用：用于估算总训练时长。

- `--gpu-vram-gb`
  - 单张 GPU 的显存大小，单位 GB。
  - 默认值：`24.0`
  - 作用：用于计算需要多少张 GPU 才能容纳估算出来的总显存。

- `--precision`
  - 权重/计算精度。
  - 可选值：`fp16`、`bf16`、`int8`、`int4`
  - 作用：影响权重显存占用，低精度通常更省显存。

- `--optimizer`
  - 优化器类型。
  - 可选值：`adam`、`adamw`、`adafactor`
  - 作用：影响优化器状态显存大小。

- `--zero-stage`
  - DeepSpeed ZeRO 分片等级。
  - 可选值：`0`、`1`、`2`、`3`
  - 作用：数值越高，参数/梯度/优化器状态在多卡之间的切分越强，单卡显存压力越小。

- `--fsdp`
  - 是否启用 FSDP（Fully Sharded Data Parallel）。
  - 作用：开启后会进一步分片参数、梯度和优化器状态，降低每张卡的显存占用。

- `--gradient-checkpointing`
  - 是否启用 gradient checkpointing。
  - 作用：减少激活缓存，从而降低显存占用，但通常会增加计算开销。

- `--json`
  - 是否以 JSON 格式输出结果。
  - 作用：便于脚本处理或保存结果。

#### 不同 method 的适用场景

- `inference`
  - 纯推理场景，不训练参数。
  - 适合做部署容量预估。

- `full`
  - 全参数微调。
  - 显存开销最大，但灵活度高，适合小中规模模型的完整训练。

- `lora`
  - 低秩适配微调。
  - 常用于在较小显存下微调大模型，是最常见的参数高效训练方式之一。

- `qlora`
  - 在低比特量化权重上做 LoRA。
  - 进一步降低权重显存占用，适合显存紧张但仍想微调大模型的场景。

- `prompt_tuning`
  - 只学习可训练 prompt token。
  - 训练参数更少，开销更低，适合任务适配和快速实验。

- `prefix_tuning`
  - 在每层注意力前加入可学习前缀向量。
  - 通常比 prompt tuning 更强一些，但训练开销也略高。

#### 输出字段说明

- `model`
  - 当前参与估算的模型信息。
  - 字段说明：
    - `name`：模型名称
    - `parameters_billion`：参数量，单位十亿
    - `hidden_size`：隐藏层维度
    - `layers`：层数
    - `seq_len`：缓存中的默认上下文长度

- `method`
  - 当前采用的训练/推理策略。
  - 字段说明：
    - `name`：方法名
    - `weight_bits`：权重位宽
    - `trainable_ratio`：可训练参数占比
    - `optimizer_state_multiplier`：优化器状态倍数
    - `gradient_multiplier`：梯度显存倍数
    - `activation_multiplier`：激活显存倍数
    - `checkpointing_factor`：checkpointing 相关缩放系数
    - `token_throughput_factor`：吞吐缩放系数

- `batch_size`
  - 本次估算使用的 batch size。

- `seq_len`
  - 本次估算使用的输入长度。

- `steps`
  - 本次估算使用的训练步数。

- `gpu_vram_gb`
  - 单卡显存大小，单位 GB。

- `estimation`
  - 估算结果主体。
  - 字段说明：
    - `weights_gb`：模型权重占用显存
    - `gradients_gb`：梯度占用显存
    - `optimizer_gb`：优化器状态占用显存
    - `activations_gb`：激活占用显存
    - `total_gb`：单卡总显存需求估算
    - `precision`：使用的精度
    - `optimizer`：使用的优化器
    - `zero_stage`：ZeRO 等级
    - `fsdp`：是否启用 FSDP
    - `gradient_checkpointing`：是否启用 gradient checkpointing
    - `sharding_factor`：当前分片缩放因子
    - `gpus`：估算所需 GPU 数量
    - `training_hours`：估算训练总时长（小时）

估算输出示例（格式会随参数不同而变化）：

| 字段 | 示例值 | 说明 |
| --- | --- | --- |
| `model.name` | `meta-llama/Llama-3.1-8B-Instruct` | 模型全称或缓存命中的模型名 |
| `model.parameters_billion` | `8.0` | 模型参数量，单位十亿 |
| `model.hidden_size` | `4096` | 隐藏层维度 |
| `model.layers` | `32` | Transformer 层数 |
| `model.seq_len` | `4096` | 缓存中的默认上下文长度 |
| `method.name` | `lora` | 当前使用的方法 |
| `method.weight_bits` | `16` | 权重位宽 |
| `method.trainable_ratio` | `0.02` | 可训练参数占比 |
| `method.optimizer_state_multiplier` | `2.0` | 优化器状态显存倍数 |
| `method.gradient_multiplier` | `0.1` | 梯度显存倍率 |
| `method.activation_multiplier` | `0.7` | 激活显存倍率 |
| `method.checkpointing_factor` | `0.85` | checkpointing 相关缩放系数 |
| `method.token_throughput_factor` | `0.9` | 吞吐缩放系数 |
| `batch_size` | `1` | 本次估算的 batch size |
| `seq_len` | `4096` | 本次估算的输入长度 |
| `steps` | `10000` | 训练步数 |
| `gpu_vram_gb` | `24.0` | 单卡显存大小，单位 GB |
| `estimation.weights_gb` | `14.9` | 模型权重占用显存 |
| `estimation.gradients_gb` | `0.1` | 梯度占用显存 |
| `estimation.optimizer_gb` | `0.3` | 优化器状态占用显存 |
| `estimation.activations_gb` | `1.7` | 激活占用显存 |
| `estimation.total_gb` | `17.0` | 单卡总显存需求估算 |
| `estimation.precision` | `bf16` | 使用的精度 |
| `estimation.optimizer` | `adamw` | 使用的优化器 |
| `estimation.zero_stage` | `0` | ZeRO 等级 |
| `estimation.fsdp` | `false` | 是否启用 FSDP |
| `estimation.gradient_checkpointing` | `true` | 是否启用 gradient checkpointing |
| `estimation.sharding_factor` | `1.0` | 当前分片缩放因子 |
| `estimation.gpus` | `1` | 估算所需 GPU 数量 |
| `estimation.training_hours` | `5.2` | 估算训练总时长（小时） |

> 提示：上表中的数值仅用于说明字段含义，实际结果会根据模型、方法和参数变化。

## 常见问题

### 什么是 gated repo？

gated repo 指的是受访问控制的 Hugging Face 仓库。模型页面虽然可见，但不能匿名直接下载，通常需要你：

- 登录 Hugging Face
- 在模型页面点击 `Request access` / `Agree and access`
- 同意许可协议，或等待仓库作者批准

### Hugging Face 的访问权限在哪里设置？

通常在模型页面本身设置，而不是在本地代码里设置。你可以在模型主页完成申请或同意协议，例如：

- `meta-llama/Llama-3.1-8B-Instruct`

如果需要本地访问，还可以在命令行登录：

```bash
hf auth login
```

然后粘贴你的 Hugging Face token。生成 token 的位置一般在 Hugging Face 网站的：

- 头像菜单
- `Settings`
- `Access Tokens`

如果只是下载模型，一般使用 `read` 权限就足够了。

### 常见报错

- `401 Unauthorized`
  - 通常表示你没有访问该模型的权限，或者还没有登录。
- `GatedRepoError`
  - 通常表示该模型是 gated repo，你需要先申请访问或同意协议。

## 说明

本仓库支持从 Hugging Face 自动读取公开模型配置，并写入本地缓存 `data/model_cache.json`。
如果模型配置中缺少参数量，仍然可以通过命令行手动补充。

估算结果适合做方案对比和容量规划，不替代真实 profiling。
