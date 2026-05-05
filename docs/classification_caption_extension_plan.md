# Classification / Caption 扩展实验重评估

## 1. 结论先行

这次重评估以 `phasewin` 当前仓库自身的实现和结果为准，不再把历史外部脚本组织直接平移进来。

当前仓库已经完成了两条主线闭环：

- classification：`CLIP ViT-L/14 + ImageNet`
- caption：`Qwen2.5-VL-3B + COCO caption token attribution`

因此扩展实验的目标不是“补主流程”，而是沿着现有 unified runtime 往外扩：

1. 在 classification 上补一个真正可比的卷积 backbone 对照
2. 在 classification 上补正确 / 错误样本拆分实验
3. 在 caption 上补一个 repo-native 的更大模型
4. baseline 扩展优先考虑是否能纳入当前仓库闭环，而不是先追求和上游脚本一模一样

基于这一点，第一轮最推荐的补实验顺序是：

1. `CLIP RN101` 正确样本主表
2. `CLIP ViT-L/14` mistake `cause / repair`
3. `CLIP RN101` mistake `cause / repair`
4. `Qwen2.5-VL-7B` native 主表

`torchvision ResNet-101` 可以做，但在 `phasewin` 里应视为辅助对照，不应和 `CLIP RN101` 处于同一优先级。
`IGOS++` 现在已经有 repo-native 实现，但现阶段仍不应作为第一轮主线。

## 2. 当前仓库已经具备的能力

### 2.1 Classification 主闭环

核心入口：

- `tasks/classification/clip_imagenet.py`

当前已经完成的主结果：

- 结果根目录：`classification_results/imagenet-clip-vitl`
- 历史主结果样本数：`4891`
- 已完成方法：
  - `greedy`
  - `phasewin`
  - `drise`
  - `dhsic`
  - `gradient`
  - `grad_eclip`
  - `ig2`

这条线已经证明 `phasewin` 的 classification 闭环是完整的：

- 统一任务入口
- 统一 `run tag`
- 统一 `json/npy` 结果格式
- 统一 `eval_classification.py` 评测

更重要的是，`clip_imagenet.py` 本身已经把 backbone 切换暴露成参数：

- `--clip-type`
- `--semantic-features`

补充说明：

- 当前 canonical classification 列表已经切到
  `datasets/imagenet/generated/*.txt`。
- 这些列表由官方 `50000`-image val GT 重新构建，固定 seed `0`，
  每个模型累计到 `5000 true / 2000 false`。
- 因此如果重跑主表，新的样本基数会和现有历史结果不同。

这意味着补 `CLIP RN101` 时不需要新建第二套 classification 主流程，只需要换模型资产和结果根目录。

### 2.2 Caption 主闭环

核心入口：

- `tasks/caption_vqa/qwen25vl_coco_caption.py`

当前已经完成的主结果：

- 结果根目录：`caption_results/Qwen2.5-VL-3B-coco-caption`
- 样本数：`275`
- 已完成方法：
  - `greedy`
  - `phasewin`
  - `drise`
  - `gradient`
  - `llavacam`

这条线同样已经闭环：

- 统一 token attribution task
- 统一结果格式
- 统一 `eval_caption.py`

并且 caption 线现在已经包含一个 repo-native 的 `IGOS++` 风格优化 baseline。

并且当前 caption 入口已经天然支持模型切换：

- `--model-name`
- `--eval-list`

再配合：

- `scripts/build_caption_eval_list.py`

说明补 `Qwen2.5-VL-7B` 的主方法实验，本质上是“沿用当前主流程换模型”，而不是回退到模型专用目录。

### 2.3 已经存在的补实验资产

当前仓库里，和第一轮补实验直接相关的文件已经有：

- classification 卷积 / pure CNN
  - `tasks/classification/torchvision_imagenet.py`
  - `scripts/run_classification_clip_rn101.sh`
  - `scripts/run_classification_resnet101.sh`
  - `scripts/build_imagenet_eval_lists.py`
  - `scripts/run_classification_mistake.sh`
- caption 模型扩展
  - `scripts/build_caption_eval_list.py`
  - `scripts/run_caption_qwen7b.sh`
这说明补实验的工程重心不该是“再造入口”，而该是：

- 明确哪些 track 属于 repo-native
- 明确哪些 baseline 值得放进第一轮主线
- 按这个边界来安排 round1 的优先级

## 3. 当前结果对扩展方案的约束

### 3.1 当前主结论已经很清楚

从现有结果看：

- classification 上：`greedy > phasewin >> 其他方法`
- caption 上：`greedy > phasewin >> 其他方法`

更具体地说：

- classification
  - `greedy`：`Insertion AUC = 0.8120`
  - `phasewin`：`Insertion AUC = 0.7835`
  - `phasewin` 平均前向调用 `940.65`
  - `greedy` 平均前向调用 `1819.74`
- caption
  - `greedy`：`Insertion AUC = 0.6405`
  - `phasewin`：`Insertion AUC = 0.6251`
  - `phasewin` 平均前向调用 `1530.73`
  - `greedy` 平均前向调用 `4296.64`

所以第一轮扩展更应该回答的是：

1. 这些结论在卷积 backbone 上是否仍然成立
2. 它们在错误样本上是否仍然成立
3. 它们在更大 Qwen 模型上是否仍然成立

而不是先继续扩大 baseline 名单。

### 3.2 `phasewin` 下的“可比性”比“上游完整复刻”更重要

当前仓库所有主结果都共享：

- 同一个 runtime
- 同一种 region replay 评测
- 同一种结果目录结构

所以扩展方案应该优先选择“仍然能进入这个闭环”的实验。

这会直接影响优先级判断：

- `CLIP RN101`：高优先级
  因为它还能走 `clip_imagenet.py`
- `Qwen2.5-VL-7B`：高优先级
  因为它还能走 `qwen25vl_coco_caption.py`
- `torchvision ResNet-101`：中优先级
  因为虽然已接入，但方法覆盖比 CLIP 线窄
- `IGOS++`：中低优先级
  因为虽然已经 repo-native，但它不是当前 caption 第一优先级 headline track

## 4. 对第一轮扩展的重新评估

### 4.1 Classification：主线优先补 `CLIP RN101`

这是第一轮最重要的 classification 扩展。

原因来自当前公开 CLIP 任务线本身：

1. 在 `phasewin` 当前实现中，它仍然沿用同一个 `clip_imagenet.py`
2. 它与当前 `CLIP ViT-L/14` 保持同一 zero-shot CLIP 任务定义
3. 它能最大程度保证表格横向可比

推荐主表方法：

- `greedy`
- `phasewin`
- `drise`
- `dhsic`
- `gradient`
- `ig2`

不建议把 `grad_eclip` 放进 RN101 主表，因为它是 ViT 结构特化方法。

直接复用的资产：

- `datasets/imagenet/generated/clip_rn101_true.txt`
- `datasets/imagenet/generated/clip_rn101_false_gt.txt`
- `datasets/imagenet/generated/clip_rn101_false_pred.txt`
- `ckpt/semantic_features/clip_rn101_imagenet_zeroweights.pt`
- `scripts/run_classification_clip_rn101.sh`

### 4.2 Classification：mistake 实验应以现有 AUC 闭环先落地

错误样本实验仍然值得做，但第一轮不建议一开始就引入一整套新 correction 指标。

更稳的顺序是：

1. 先把 `cause / repair` 两套 eval list 跑出来
2. 先用现有 `eval_classification.py` 做统一 AUC 汇总
3. 看结论是否已经足够明显
4. 只有在 AUC 无法区分时，再加 correction-style 指标

在 `phasewin` 当前仓库里，这条线已经有很好的落脚点：

- `scripts/build_imagenet_eval_lists.py`
- `scripts/run_classification_mistake.sh`

这意味着 mistake 设计不需要另起炉灶，只需要明确实验解释：

- `cause`
  - 用 `*_false_pred.txt`
  - target 是错误预测类
- `repair`
  - 用 `*_false_gt.txt`
  - target 是 GT 类

第一轮建议的顺序：

1. `CLIP ViT-L/14` mistake
2. `CLIP RN101` mistake
3. `ResNet-101` mistake 作为补充对照

### 4.3 Pure CNN：`ResNet-101` 适合做辅助对照，不适合做 headline

当前仓库已经有：

- `tasks/classification/torchvision_imagenet.py`
- `scripts/run_classification_resnet101.sh`

这说明 pure CNN 不是做不了，而是研究定位需要更谨慎。

因为一旦切到 pure torchvision 分类器，变化的不只是 backbone：

- 分类头变了
- 训练范式变了
- zero-shot 语义匹配消失了
- 方法覆盖也变少了

当前 pure CNN 线只支持：

- `greedy`
- `phasewin`
- `drise`
- `gradient`

它缺少：

- `dhsic`
- `ig2`
- `grad_eclip`

所以它更适合作为：

- “CLIP 卷积线之外，再给一个纯分类 CNN 控制组”

而不是和 `CLIP RN101` 并列成第一条核心扩展主线。

### 4.4 Caption：第一轮优先补 `Qwen2.5-VL-7B`，不是先补 baseline

对 caption 而言，当前最自然的扩展仍然不是先扩大 baseline 名单，而是先完成 `Qwen2.5-VL-7B`。

原因完全来自当前仓库本身：

1. `qwen25vl_coco_caption.py` 已经支持 `--model-name`
2. `build_caption_eval_list.py` 已经能重建模型相关 token metadata
3. `run_caption_qwen7b.sh` 已经把主方法实验入口搭好了

这条线是 repo-native 的、可评测的、可与 3B 主表直接对照的。

因此第一轮 caption 扩展应优先做：

1. 生成 `Qwen2.5-VL-7B` 的 eval list
2. 跑 `greedy / phasewin / drise / gradient / llavacam / igos_pp`
3. 用 `eval_caption.py` 汇总

### 4.5 Caption baseline：`IGOS++` 已经 native，但仍应视为第二梯队

`IGOS++` 现在已经接入当前仓库，但仍不应被当作 caption 第一优先级主线。

原因不是工程接入问题，而是研究优先级：

- 当前更需要先验证模型规模扩展是否延续已有结论
- `Qwen2.5-VL-7B` 和 `Qwen2.5-VL-3B` 的横向可比性更高
- `IGOS++` 更适合作为补充 baseline，而不是 headline track

也就是说，它现在已经是统一 runtime 的原生一员，但在报告优先级上仍应靠后。

因此更合理的定位是：

- 第一轮先完成 `Qwen2.5-VL-7B` native 主表
- 第二步再决定是否要把 `IGOS++` 提升为 headline 比较对象

如果后面要继续推进 `IGOS++`，更推荐的方向是：

- 做默认超参数的稳定性评估
- 在 `3B / 7B` 上补充统一结果汇总
- 和 `gradient / llavacam / phasewin` 做质量与效率对照

## 5. 数据集是否要加

第一轮不建议加新数据集。

当前仓库的主结果、补实验脚本和评测逻辑都围绕：

- classification：ImageNet
- caption：COCO

而你现在想回答的问题主要是：

- backbone 差异
- 正确 / 错误样本差异
- 模型规模差异
- baseline 是否值得保留

这些问题都能在现有数据上回答。

如果一开始就引入 `CUB-200-2011` 或 `Stanford Cars`，会同时引入：

- 新标签空间
- 新 prompt / eval list
- 新错误类型
- 新数据粒度偏置

这会削弱第一轮结论的解释性。

所以 round1 保持：

- `ImageNet`
- `COCO`

最合适。

## 6. 建议的第一轮执行顺序

### 6.1 必做

1. `CLIP RN101` 正确样本主表
   - 脚本：`scripts/run_classification_clip_rn101.sh`
2. `CLIP ViT-L/14` mistake `cause / repair`
   - 脚本：`scripts/run_classification_mistake.sh --model clip_vitl14`
3. `Qwen2.5-VL-7B` 主表
   - 先跑：`scripts/build_caption_eval_list.py`
   - 再跑：`scripts/run_caption_qwen7b.sh`

### 6.1.1 对应脚本

- 一键入口
  - `scripts/run_round1_extension.sh`
- `CLIP RN101` 主表
  - `scripts/run_classification_clip_rn101.sh`
- `CLIP ViT-L/14` mistake
  - `scripts/run_classification_clip_vitl_mistake.sh`
- `CLIP RN101` mistake
  - `scripts/run_classification_clip_rn101_mistake.sh`
- 通用 mistake 构建与运行
  - `scripts/run_classification_mistake.sh`
  - `scripts/build_imagenet_eval_lists.py`
- `Qwen2.5-VL-7B` 主表
  - `scripts/run_caption_qwen7b.sh`
  - `scripts/build_caption_eval_list.py`

### 6.2 有余力再做

4. `CLIP RN101` mistake `cause / repair`
5. `torchvision ResNet-101` 正确 / 错误样本对照

### 6.3 暂不作为 round1 主线

6. `IGOS++`
7. `TAM`
8. 新数据集

## 7. 最终建议

如果目标是“少量但完整、而且真正贴着当前仓库”的补实验，最合理的 round1 方案是：

- classification
  - `CLIP RN101` 正确样本
  - `CLIP ViT-L/14` mistake
  - `CLIP RN101` mistake
  - `ResNet-101` 只做辅助对照
- caption
  - `Qwen2.5-VL-7B` native 主表
  - `IGOS++` 作为 repo-native 补充 baseline，可跑，但暂不放进主线 headline 表

这套方案会借鉴既有实验思路，但组织方式、优先级和落地路径都以 `phasewin` 当前代码结构和结果闭环为中心。
