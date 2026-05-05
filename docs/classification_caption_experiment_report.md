# Classification + Caption 实验报告

## 1. 报告范围

本报告基于仓库中已经完成并落盘的两组实验结果整理而成：

- `classification_results/imagenet-clip-vitl/`
- `caption_results/Qwen2.5-VL-3B-coco-caption/`

结果汇总直接读取以下文件：

- `classification_results/imagenet-clip-vitl/eval_summary.json`
- `caption_results/Qwen2.5-VL-3B-coco-caption/eval_summary.json`

本报告只覆盖 `classification` 和 `caption` 两个任务，不包含 detection。

## 1.1 当前工作区完成度

当前 `phasewin` 工作区里，真正已经完成并可直接汇总的主结果只有两组：

- `classification_results/imagenet-clip-vitl/`
- `caption_results/Qwen2.5-VL-3B-coco-caption/`

按结果目录里的 `json/` 与 `npy/` 文件数统计：

- 下列 classification 计数对应历史运行快照。
- 当前 canonical classification 列表已经切到
  `datasets/imagenet/generated/clip_vitl14_true.txt`，样本数固定为 `5000`。
- 因此重跑不会再得到这里的 `4891 / 4891` 历史计数。

- classification
  - `dhsic-slico-division-50-batch-32-tf-cpu`: `4891 / 4891`
  - `drise-slico-division-50-1.0-1.0-nmasks-1000`: `4891 / 4891`
  - `grad_eclip-slico-division-50-1.0-1.0`: `4891 / 4891`
  - `gradient-slico-division-50-1.0-1.0`: `4891 / 4891`
  - `greedy-slico-division-50-1.0-1.0`: `4891 / 4891`
  - `ig2-slico-division-50-1.0-1.0`: `4891 / 4891`
  - `phasewin-slico-division-50-1.0-1.0-window-16`: `4891 / 4891`
- caption
  - `drise-slico-division-64-1.0-1.0-nmasks-1000`: `275 / 275`
  - `gradient-slico-division-64-1.0-1.0`: `275 / 275`
  - `greedy-slico-division-64-1.0-1.0`: `275 / 275`
  - `llavacam-slico-division-64-1.0-1.0`: `275 / 275`
  - `phasewin-slico-division-64-1.0-1.0-window-16`: `275 / 275`

其中 classification 目录下还存在一个旧的空目录：

- `classification_results/imagenet-clip-vitl/dhsic-slico-division-50-batch-32/`

它没有实际结果文件，本报告不把它计入完成实验。

截至当前文档版本，以下补实验相关目录或脚本虽然已经存在，但还不属于“已完成主结果”：

- `classification_results/imagenet-clip-rn101/`
- `classification_results/imagenet-resnet101/`
- `classification_results/*-mistake/`
- `caption_results/Qwen2.5-VL-7B-coco-caption/`

## 2. 统一实验协议

两类任务都采用“先得到区域重要性排序，再做插入/删除重放评测”的统一协议。

- 区域划分：默认都使用 `superpixel`，算法为 `slico`
- 搜索类方法的目标分数：
  `gain = lambda1 * insertion_score + lambda2 * (1 - deletion_score)`
- 本次实验中 `lambda1 = 1.0`，`lambda2 = 1.0`
- 每个 run 都保存为统一结构：
  `<result-root>/<run-tag>/{json,npy}/`

评测指标来自 `attribution_research/evaluation/auc_faithfulness.py`：

- `Insertion AUC`：越大越好，表示按解释顺序逐步插入区域时，目标分数上升越快
- `Deletion AUC`：越小越好，表示按解释顺序逐步删除区域时，目标分数下降越快
- `Average highest`：插入曲线上的最高分，越大越好
- `@30% area` / `@50% area`：当可见区域不超过 30% / 50% 时，插入曲线能达到的最高分，越大越好
- `Avg model forward calls`：平均等价单图前向调用次数，越小越好

caption 任务额外计算词敏感性指标，阈值为 `0.2`：

- `Insertion sensitivity AUC`：越大越好
- `Deletion sensitivity AUC`：越小越好

说明：

- 对 `gradient`、`grad_eclip`、`ig2`、`llavacam` 这类 map-based 方法，当前汇总里没有统一记录 `average_model_forward_calls`，因此效率对比主要在搜索类、D-RISE 和 D-HSIC 之间进行。

## 3. 方法说明

### 3.1 搜索类方法

- `greedy`
  每一步都对“所有尚未被选中的区域”逐个计算边际增益，选出当前最优区域再继续，属于标准黑盒贪心搜索，代价高但通常效果最好。
- `phasewin`
  以贪心为基础，引入 phase-window 机制，只保留高收益候选并在窗口内选择，目标是在尽量保持排序质量的同时显著减少前向调用。
- `drise`
  采样大量随机平滑 mask，按目标分数对 mask 加权累加，先得到像素级 saliency map，再聚合到 superpixel 上形成区域排序。
- `dhsic`
  classification 专用方法，调用 `xplique.HsicAttributionMethod` 生成 D-HSIC saliency map，再做区域回放评测；本次配置中 TensorFlow 被限制在 CPU 上运行。

### 3.2 梯度 / map-replay 类方法

- `gradient`
  直接对目标分数做输入梯度，使用 `|grad * input|` 形成 saliency map，然后按 superpixel 平均聚合成区域排序。
- `grad_eclip`
  classification 专用方法，针对 CLIP ViT 的后几层注意力块，结合类 token 梯度、value 特征和 `q-k` 相似度构造 dense saliency。
- `ig2`
  classification 专用方法，基于多尺度高斯模糊参考图构造路径，并通过表示空间约束与梯度积分近似生成 saliency。
- `llavacam`
  caption 专用方法，在 Qwen2.5-VL 的语言解码层上注册 hook，提取视觉 token 特征与梯度，构造类似 CAM 的热力图。

说明：

- `gradient`、`grad_eclip`、`ig2`、`llavacam` 在本仓库里都不是直接输出区域顺序，而是先输出二维 saliency map，再统一转换成区域排序并复用同一套 insertion/deletion 评测。

## 4. Classification 实验

### 4.1 实验设置

- 任务：ImageNet 图像分类解释
- 数据集目录：`datasets/imagenet/ILSVRC2012_img_val`
- 评测列表：历史快照使用旧 true list；当前 canonical 列表是 `datasets/imagenet/generated/clip_vitl14_true.txt`
- 本报告对应的历史评测样本数：`4891`
- 当前 canonical 样本数：`5000`
- 模型：`CLIP ViT-L/14`
- 文本侧分类权重：`ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt`
- 图像预处理：任务脚本中统一 resize 到 `224 x 224`
- 分割设置：`superpixel + slico + division_number=50`

本次实际完成的 classification 方法：

- `greedy`
- `phasewin`
- `drise`
- `dhsic`
- `gradient`
- `grad_eclip`
- `ig2`

关键超参数：

- `phasewin`: `window_size=16`
- `drise`: `n_masks=1000`, `grid=16x16`, `prob_thresh=0.5`
- `dhsic`: `batch_size=32`, `tf_device=cpu`
- `gradient`: `score_mode=prob`
- `grad_eclip`: `layer_span=1`
- `ig2`: `steps=32`, `step_size=8.0`, `blur_sigmas=[3,7,15,31]`

### 4.2 结果总表

| 方法 | 家族 | Insertion AUC | Deletion AUC | Highest | @30% area | @50% area | Avg Fwd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | search | 0.8120 | 0.1385 | 0.9646 | 0.8675 | 0.9456 | 1819.74 |
| phasewin | search | 0.7835 | 0.1669 | 0.9576 | 0.8242 | 0.9186 | 940.65 |
| dhsic | search/map | 0.6617 | 0.2684 | 0.9028 | 0.7417 | 0.8350 | 598.00 |
| grad_eclip | gradient | 0.6370 | 0.2849 | 0.9124 | 0.6736 | 0.8072 | - |
| drise | search/map | 0.6212 | 0.3091 | 0.9126 | 0.5901 | 0.8040 | 2081.69 |
| gradient | gradient | 0.4302 | 0.4692 | 0.8892 | 0.2121 | 0.4920 | - |
| ig2 | gradient | 0.4072 | 0.4955 | 0.8890 | 0.1863 | 0.4574 | - |

本报告中的所有方法都在 `4891` 个样本上完成评测，对应每个 run 目录下均有 `4891` 个 JSON 结果文件。当前 canonical eval list 已更新为 `5000` 条，因此复现实验时样本数会不同。

### 4.3 结果解读

- `greedy` 在所有主指标上都是最优，说明穷举式区域贪心搜索在 classification 任务上仍然是当前最强基线。
- `phasewin` 是最均衡的方法。它的 `Insertion AUC` 只比 `greedy` 低 `0.0286`，但平均前向调用从 `1819.74` 降到 `940.65`，下降约 `48.3%`。
- `dhsic` 是当前 classification 里最强的非贪心 map-based 基线。它的效果明显强于普通 `gradient` 和 `ig2`，而且记录到的平均前向调用只有 `598.00`。
- `grad_eclip` 在梯度家族中表现最好，特别是在 `@30% area` 和 `@50% area` 上明显强于普通输入梯度，说明它对 CLIP ViT 后层结构的专门适配是有效的。
- `drise` 的表现不差，但在当前设置下“更慢且更弱”：`Avg Fwd=2081.69` 高于 `greedy` 与 `phasewin`，而 AUC 又低于这两种搜索法。
- `gradient` 和 `ig2` 明显落后，尤其在小面积可见区域下的最高分很低，说明它们生成的 saliency 排序在 region replay 评测下不够集中。

### 4.4 Classification 小结

- 如果只追求最强效果，当前最佳方法是 `greedy`。
- 如果更看重效果与效率折中，`phasewin` 是最推荐的选择。
- 如果希望使用 map-based 方法，当前优先级大致可写为：`dhsic >= grad_eclip > drise >> gradient > ig2`。

## 5. Caption 实验

### 5.1 实验设置

- 任务：Qwen2.5-VL caption token attribution
- 数据集目录：`datasets/coco/val2017`
- 评测列表：`datasets/Qwen2.5-VL-3B-coco-caption.json`
- 评测样本数：`275`
- 模型：`model_checkpoint/Qwen2.5-VL-3B-Instruct`
- 分割设置：`superpixel + slico + division_number=64`

这组实验解释的不是 GT caption，也不是整句统一得分，而是对评测列表中预先选中的生成 token / word 序列做归因。每条样本都包含：

- `generate_sentence`
- `generated_ids`
- `selected_interpretation_token_id`
- `selected_interpretation_token_word_id`
- `words`

本次实际完成的 caption 方法：

- `greedy`
- `phasewin`
- `drise`
- `gradient`
- `llavacam`

关键超参数：

- `phasewin`: `window_size=16`
- `drise`: `n_masks=1000`, `grid=16x16`, `prob_thresh=0.5`
- `llavacam`: `layer_index=32`

### 5.2 结果总表

| 方法 | 家族 | Insertion AUC | Deletion AUC | SensIns | SensDel | Highest | @30% area | Avg Fwd |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | search | 0.6405 | 0.4372 | 0.5946 | 0.2858 | 0.6951 | 0.6507 | 4296.64 |
| phasewin | search | 0.6251 | 0.4522 | 0.5736 | 0.3052 | 0.6835 | 0.6333 | 1530.73 |
| drise | search/map | 0.5608 | 0.5087 | 0.4771 | 0.3893 | 0.6645 | 0.5307 | 2128.02 |
| gradient | gradient | 0.5365 | 0.5315 | 0.4397 | 0.4298 | 0.6615 | 0.4992 | - |
| llavacam | gradient | 0.5248 | 0.5460 | 0.4184 | 0.4530 | 0.6599 | 0.4917 | - |

所有方法都在 `275` 个样本上完成评测，对应每个 run 目录下均有 `275` 个 JSON 结果文件。

### 5.3 结果解读

- `greedy` 依然是 caption 任务上的最优方法，既拿到了最高的 `Insertion AUC`，又有最小的 `Deletion AUC`，在词敏感性指标上也同样最好。
- `phasewin` 的效果非常接近 `greedy`，但平均前向调用从 `4296.64` 降到 `1530.73`，下降约 `64.4%`，是当前 caption 实验里最有性价比的方法。
- `drise` 处在中间水平，性能明显好于两种梯度法，但仍弱于两种搜索法。
- 在当前实现和数据设置下，`gradient` 略好于 `llavacam`；也就是说，Qwen 上的 hook-based CAM 变体还没有体现出相对普通输入梯度的优势。
- caption 结果应该理解为“对选定生成词序列的视觉依据是否能被正确排序并回放出来”，而不是 caption 文本质量本身的比较。

### 5.4 Caption 小结

- 如果只看解释质量，当前最佳方法仍然是 `greedy`。
- 如果看质量与成本平衡，`phasewin` 是最值得优先保留的方案。
- 在当前 Qwen2.5-VL caption token attribution 设置下，搜索类方法整体明显优于梯度类方法。

## 6. 综合结论

- 两个任务上都出现了相同趋势：`greedy > phasewin >> 其他方法`。
- `phasewin` 是本次实验最稳定的折中方案：classification 中前向调用减少约 `48.3%`，caption 中减少约 `64.4%`，同时仍保持接近 `greedy` 的 AUC。
- `drise` 在两个任务上都没有打赢 `phasewin`，并且其前向调用也不占优势，因此当前不属于最推荐方案。
- classification 中，`dhsic` 和 `grad_eclip` 证明了“面向 CLIP 的专用 map-based 方法”比通用输入梯度更有效。
- caption 中，Qwen2.5-VL 的 `llavacam` 还没有超过普通输入梯度，说明这一路线仍需要进一步调参或改实现。

## 7. 建议的后续动作

- 后续如只保留少量主方法，建议至少保留 `greedy`、`phasewin`，classification 里再保留 `dhsic` 或 `grad_eclip` 作为 map-based 对照。
- 如果要补齐论文式对比，classification 可以继续补充公开 baseline，并优先保证各 backbone 的评测协议一致。
- 如果要继续优化 caption 侧效率，优先从 `phasewin` 出发，而不是继续扩大 `drise` 的随机 mask 数量。
