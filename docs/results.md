# Results Summary

整理规则：每个 `eval_summary.json` 生成一张独立表；仅保留所有文件都存在的公共指标（移除公共不存在的指标，如 `mufidelity_config`）。
行按方法名排序，数值保留 4 位小数，空值显示为 `N/A`。

## `results/caption/Qwen2.5-VL-3B-coco-caption/eval_summary.json`
共 6 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drise-slico-division-64-1.0-1.0-nmasks-1000 | 0.5608 | 0.5087 | N/A | N/A | N/A | N/A | 0.4771 | 0.3893 | 0.6600 | 0.6645 | 0.5307 | 0.5928 | 2128.0218 | N/A | N/A | N/A | N/A | 275 |
| gradient-slico-division-64-1.0-1.0 | 0.5365 | 0.5315 | N/A | N/A | N/A | N/A | 0.4397 | 0.4298 | 0.6575 | 0.6615 | 0.4992 | 0.5551 | N/A | N/A | N/A | N/A | N/A | 275 |
| greedy-slico-division-64-1.0-1.0 | 0.6405 | 0.4372 | N/A | N/A | N/A | N/A | 0.5946 | 0.2858 | 0.6908 | 0.6951 | 0.6507 | 0.6783 | 4296.6400 | N/A | N/A | N/A | N/A | 275 |
| igos_pp-slico-division-64-1.0-1.0 | 0.5376 | 0.5296 | N/A | N/A | N/A | N/A | 0.4388 | 0.4266 | 0.6574 | 0.6620 | 0.4945 | 0.5540 | 49.0000 | 106.1382 | 155.1382 | N/A | N/A | 275 |
| llavacam-slico-division-64-1.0-1.0 | 0.5248 | 0.5460 | N/A | N/A | N/A | N/A | 0.4184 | 0.4530 | 0.6547 | 0.6599 | 0.4917 | 0.5401 | N/A | N/A | N/A | N/A | N/A | 275 |
| phasewin-slico-division-64-1.0-1.0-window-16 | 0.6251 | 0.4522 | N/A | N/A | N/A | N/A | 0.5736 | 0.3052 | 0.6786 | 0.6835 | 0.6333 | 0.6618 | 1530.7345 | N/A | N/A | N/A | N/A | 275 |

## `results/caption/Qwen2.5-VL-7B-coco-caption/eval_summary.json`
共 6 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drise-slico-division-64-1.0-1.0-nmasks-1000 | 0.5540 | 0.5019 | N/A | N/A | N/A | N/A | 0.4610 | 0.3771 | 0.6868 | 0.6821 | 0.5046 | 0.5831 | 2000.0000 | 106.1382 | 2106.1382 | N/A | N/A | 275 |
| gradient-slico-division-64-1.0-1.0 | 0.5279 | 0.5248 | N/A | N/A | N/A | N/A | 0.4210 | 0.4149 | 0.6842 | 0.6791 | 0.4746 | 0.5425 | N/A | N/A | N/A | N/A | N/A | 275 |
| greedy-slico-division-64-1.0-1.0 | 0.6284 | 0.4350 | N/A | N/A | N/A | N/A | 0.5721 | 0.2815 | 0.7081 | 0.7064 | 0.6224 | 0.6670 | 2931.5200 | 106.1382 | 3037.6582 | N/A | N/A | 275 |
| igos_pp-slico-division-64-1.0-1.0 | 0.5350 | 0.5219 | N/A | N/A | N/A | N/A | 0.4313 | 0.4097 | 0.6874 | 0.6816 | 0.4795 | 0.5507 | 53.2600 | N/A | N/A | N/A | N/A | 275 |
| llavacam-slico-division-64-1.0-1.0 | 0.5340 | 0.5357 | N/A | N/A | N/A | N/A | 0.4347 | 0.4362 | 0.6890 | 0.6824 | 0.4724 | 0.5439 | N/A | N/A | N/A | N/A | N/A | 275 |
| phasewin-slico-division-64-1.0-1.0-window-pct-30 | 0.6155 | 0.4467 | N/A | N/A | N/A | N/A | 0.5535 | 0.2968 | 0.7015 | 0.6995 | 0.6074 | 0.6509 | 1431.6000 | 106.1382 | 1537.7382 | N/A | N/A | 275 |

## `results/classification/imagenet-clip-rn101/eval_summary.json`
共 4 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drise-slico-division-50-1.0-1.0-nmasks-1000 | 0.4627 | 0.1232 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7917 | 0.4157 | 0.5872 | 2000.0000 | 81.6476 | 2081.6476 | 0.1729 | 5000 | 5000 |
| gradient-slico-division-50-1.0-1.0 | 0.3495 | 0.2193 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7677 | 0.1902 | 0.3892 | N/A | N/A | N/A | 0.2071 | 5000 | 5000 |
| greedy-slico-division-50-1.0-1.0 | 0.6525 | 0.0650 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.8943 | 0.6072 | 0.8052 | 1736.6768 | 81.6476 | 1818.3244 | 0.2093 | 5000 | 5000 |
| phasewin-slico-division-50-1.0-1.0-window-pct-30 | 0.5981 | 0.0674 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.8783 | 0.5287 | 0.7091 | 951.1576 | 81.6476 | 1032.8052 | 0.2041 | 5000 | 5000 |

## `results/classification/imagenet-clip-rn101/mistake/cause/eval_summary.json`
共 4 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient-slico-division-50-1.0-1.0 | 0.1891 | 0.1120 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.5097 | 0.1055 | 0.2168 | N/A | N/A | N/A | N/A | N/A | 2000 |
| greedy-slico-division-50-1.0-1.0 | 0.4831 | 0.0309 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7440 | 0.4488 | 0.6297 | 1748.2620 | 81.9610 | 1830.2230 | N/A | N/A | 2000 |
| ig2-slico-division-50-1.0-1.0 | 0.1930 | 0.1150 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.5091 | 0.1078 | 0.2251 | N/A | N/A | N/A | N/A | N/A | 2000 |
| phasewin-slico-division-50-1.0-1.0-window-pct-30 | 0.4264 | 0.0316 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7175 | 0.3655 | 0.5281 | 1014.2000 | 81.9610 | 1096.1610 | N/A | N/A | 2000 |

## `results/classification/imagenet-clip-rn101/mistake/repair/eval_summary.json`
共 4 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient-slico-division-50-1.0-1.0 | 0.0605 | 0.0349 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.1849 | 0.0554 | 0.1043 | N/A | N/A | N/A | N/A | N/A | 2000 |
| greedy-slico-division-50-1.0-1.0 | 0.2747 | 0.0105 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.4802 | 0.3093 | 0.4184 | 1748.2620 | 81.9610 | 1830.2230 | N/A | N/A | 2000 |
| ig2-slico-division-50-1.0-1.0 | 0.0615 | 0.0361 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.1832 | 0.0566 | 0.1048 | N/A | N/A | N/A | N/A | N/A | 2000 |
| phasewin-slico-division-50-1.0-1.0-window-pct-30 | 0.2193 | 0.0107 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.4267 | 0.2198 | 0.3133 | 835.5070 | 81.9610 | 917.4680 | N/A | N/A | 2000 |

## `results/classification/imagenet-clip-vitl/eval_summary.json`
共 7 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dhsic-slico-division-50-batch-32-tf-cpu | 0.6755 | 0.2617 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9191 | 0.7353 | 0.8428 | 500.0000 | 81.6200 | 581.6200 | N/A | N/A | 5000 |
| drise-slico-division-50-1.0-1.0-nmasks-1000 | 0.6364 | 0.3161 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9282 | 0.6096 | 0.8166 | 2000.0000 | 81.6200 | 2081.6200 | N/A | N/A | 5000 |
| grad_eclip-slico-division-50-1.0-1.0 | 0.6488 | 0.2791 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9273 | 0.6674 | 0.8131 | N/A | N/A | N/A | N/A | N/A | 5000 |
| gradient-slico-division-50-1.0-1.0 | 0.4404 | 0.4783 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9081 | 0.2131 | 0.5013 | N/A | N/A | N/A | N/A | N/A | 5000 |
| greedy-slico-division-50-1.0-1.0 | 0.8239 | 0.1388 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9707 | 0.8787 | 0.9523 | 1735.9000 | 81.6200 | 1817.5200 | N/A | N/A | 5000 |
| ig2-slico-division-50-1.0-1.0 | 0.4213 | 0.5012 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9081 | 0.1818 | 0.4585 | N/A | N/A | N/A | N/A | N/A | 5000 |
| phasewin-slico-division-50-1.0-1.0-window-pct-30 | 0.7990 | 0.1625 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9653 | 0.8443 | 0.9300 | 871.8444 | 81.6200 | 953.4644 | N/A | N/A | 5000 |

## `results/classification/imagenet-clip-vitl/mistake/cause/eval_summary.json`
共 7 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dhsic-slico-division-50-batch-32-tf-cpu | 0.4035 | 0.1518 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7124 | 0.4608 | 0.5757 | 500.0000 | 82.1990 | 582.1990 | N/A | N/A | 2000 |
| drise-slico-division-50-1.0-1.0-nmasks-1000 | 0.4133 | 0.1605 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7474 | 0.4073 | 0.5851 | 2000.0000 | 82.1990 | 2082.1990 | N/A | N/A | 2000 |
| grad_eclip-slico-division-50-1.0-1.0 | 0.3958 | 0.1616 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7363 | 0.4273 | 0.5644 | N/A | N/A | N/A | N/A | N/A | 2000 |
| gradient-slico-division-50-1.0-1.0 | 0.2607 | 0.2831 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.6933 | 0.1396 | 0.3118 | N/A | N/A | N/A | N/A | N/A | 2000 |
| greedy-slico-division-50-1.0-1.0 | 0.6837 | 0.0652 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.8932 | 0.7492 | 0.8576 | 1755.7660 | 82.1990 | 1837.9650 | N/A | N/A | 2000 |
| ig2-slico-division-50-1.0-1.0 | 0.2474 | 0.2980 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.6913 | 0.1149 | 0.2813 | N/A | N/A | N/A | N/A | N/A | 1863 |
| phasewin-slico-division-50-1.0-1.0-window-pct-30 | 0.6421 | 0.0705 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.8774 | 0.6743 | 0.8105 | 1026.5210 | 82.1990 | 1108.7200 | N/A | N/A | 2000 |

## `results/classification/imagenet-clip-vitl/mistake/repair/eval_summary.json`
共 7 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dhsic-slico-division-50-batch-32-tf-cpu | 0.1447 | 0.0642 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.3620 | 0.2814 | 0.3227 | 500.0000 | 82.1990 | 582.1990 | N/A | N/A | 2000 |
| drise-slico-division-50-1.0-1.0-nmasks-1000 | 0.1990 | 0.0516 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.4541 | 0.2787 | 0.3774 | 2000.0000 | 82.1990 | 2082.1990 | N/A | N/A | 2000 |
| grad_eclip-slico-division-50-1.0-1.0 | 0.1832 | 0.0508 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.4475 | 0.3195 | 0.3898 | N/A | N/A | N/A | N/A | N/A | 2000 |
| gradient-slico-division-50-1.0-1.0 | 0.0996 | 0.0968 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.3251 | 0.0882 | 0.1852 | N/A | N/A | N/A | N/A | N/A | 2000 |
| greedy-slico-division-50-1.0-1.0 | 0.4827 | 0.0228 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.7425 | 0.6352 | 0.7161 | 1755.7660 | 82.1990 | 1837.9650 | N/A | N/A | 2000 |
| ig2-slico-division-50-1.0-1.0 | 0.0931 | 0.1042 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.3169 | 0.0683 | 0.1653 | N/A | N/A | N/A | N/A | N/A | 2000 |
| phasewin-slico-division-50-1.0-1.0-window-pct-30 | 0.4291 | 0.0243 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.6989 | 0.5316 | 0.6458 | 1046.4570 | 82.1990 | 1128.6560 | N/A | N/A | 2000 |

## `results/classification/imagenet-resnet101/eval_summary.json`
共 4 个方法，公共指标数：18。

| method | insertion_auc | deletion_auc | insertion_iou_auc | deletion_iou_auc | insertion_cls_auc | deletion_cls_auc | insertion_sensitivity_auc | deletion_sensitivity_auc | sensitivity_highest | average_highest | average_highest_30pct_area | average_highest_50pct_area | average_model_forward_calls | average_eval_model_forward_calls | average_total_model_forward_calls | mufidelity | mufidelity_n_samples | n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drise-slico-division-50-1.0-1.0-nmasks-1000 | 0.6124 | 0.2423 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.8270 | 0.6718 | 0.7650 | 2000.0000 | 81.9639 | 2081.9639 | N/A | N/A | 1441 |
| gradient-slico-division-50-1.0-1.0 | 0.4649 | 0.3637 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.8025 | 0.3629 | 0.5861 | N/A | N/A | N/A | N/A | N/A | 5000 |
| greedy-slico-division-50-1.0-1.0 | 0.7926 | 0.1449 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9349 | 0.8780 | 0.9238 | 1744.0680 | 81.8332 | 1825.9012 | N/A | N/A | 5000 |
| phasewin-slico-division-50-1.0-1.0-window-pct-30 | 0.7672 | 0.1556 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.9225 | 0.8519 | 0.9035 | 907.7300 | 81.8332 | 989.5632 | N/A | N/A | 5000 |
