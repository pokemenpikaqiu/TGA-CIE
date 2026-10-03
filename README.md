# TGA-CIE

Time-Granularity-Aware CfC-Inspired Interval Encoding for Temporal Knowledge Graph Entity Alignment.

本压缩包只包含训练源代码和两个基准数据集。论文、图片、日志和模型权重不在其中。

## 环境

- Python 3.10 或更高版本
- 建议使用带 CUDA 的 GPU。代码在 CUDA 上使用 bfloat16；没有 GPU 时会退回 CPU，速度会慢很多。

安装依赖：

```bash
pip install torch numpy
```

`torch` 请按本机 CUDA 版本安装，见 [https://pytorch.org](https://pytorch.org)。

## 目录

```text
TGA-CIE/
  run_train.py          # 训练入口
  tea_mmgr/             # 模型、数据读取、RotatE 先验和训练循环
  data/ICEWS05-15/      # DICEWS 使用的图，约按天计时
  data/YAGO-WIKI50K/    # 约按年计时
```

`ICEWS05-15` 里的 `sup_pairs` 是种子对，`ref_pairs` 是测试对。`--n_seed 1000` 对应 DICEWS-1K，`--n_seed 200` 对应 DICEWS-200。`YAGO-WIKI50K` 上 `--n_seed 5000` 和 `--n_seed 1000` 分别对应 5K 和 1K 划分。

## 运行

在解压后的 `TGA-CIE` 目录下执行。日志和最优权重会写到 `--work_dir`，默认不要把这些输出再放回数据包。

论文中 DICEWS-1K 的配置：

```bash
python run_train.py \
  --data_root ./data \
  --work_dir ./runs \
  --dataset ICEWS05-15 \
  --n_seed 1000 \
  --seq_len 32 \
  --n_max 64 \
  --event_sample strat \
  --time_sim idf \
  --w_time 0.25 \
  --w_glob 0.65 \
  --batch_size 96 \
  --n_neg 48 \
  --epochs 80 \
  --patience 6 \
  --boot_start 1 \
  --boot_max 800 \
  --boot_min 0.30 \
  --boot_margin 0.04 \
  --interval_gate \
  --gate_bias -3 \
  --d_pre 512 \
  --d_glob 512 \
  --d_v 512 \
  --d_h 256 \
  --rotate_epochs 40 \
  --lr 3e-4 \
  --tau 0.05 \
  --lam 0.01 \
  --freeze_prior 12 \
  --csls_k 10 \
  --ridge_lam 25 \
  --seed 0
```

也可以写成 `python -m tea_mmgr.train`，参数相同。

另外三个划分只改数据相关的参数：

- DICEWS-200：`--dataset ICEWS05-15 --n_seed 200 --seq_len 24 --n_max 48 --event_sample linspace --time_sim jaccard --w_time 0.20 --w_glob 0.50 --epochs 100 --patience 8 --ridge_lam 25 --gate_bias -3`
- YAGO-WIKI50K-5K：`--dataset YAGO-WIKI50K --n_seed 5000 --seq_len 24 --n_max 48 --event_sample linspace --time_sim jaccard --w_time 0.20 --w_glob 0.35 --batch_size 32 --chunk 256 --n_neg 24 --epochs 180 --boot_start 20 --boot_lr_mult 0.6 --patience 12 --ridge_lam 40 --gate_bias -3`
- YAGO-WIKI50K-1K：`--dataset YAGO-WIKI50K --n_seed 1000 --seq_len 24 --n_max 48 --event_sample linspace --time_sim jaccard --w_time 0.20 --w_glob 0.50 --batch_size 64 --chunk 1024 --epochs 100 --patience 8 --ridge_lam 25 --gate_bias -2`

未写出的宽度、学习率和对比学习参数与上面的 DICEWS-1K 命令相同。必须加上 `--interval_gate`，否则运行的是不带间隔使用门的对照，而不是论文报告的 Full 模型。
