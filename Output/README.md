# Patch detector results

Full variant, seed 42. Clean validation ROC-AUC selected epoch 8 of 20. All conditions use the same 1,463 test images.

| Test condition | Accuracy | Fake recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.9542 | 0.9798 | 0.9530 | 0.9699 |
| Q_90 | 0.9515 | 0.9827 | 0.9505 | 0.9691 |
| Q_70 | 0.9487 | 0.9755 | 0.9475 | 0.9670 |
| Q_50 | 0.9357 | 0.9539 | 0.9337 | 0.9656 |

The checkpoints folder holds best.pt for inference and last.pt for resuming training. The logs folder contains the original run settings, epoch history and test report. The predictions folder contains image-level fake scores for each condition. The examples folder illustrates clean-test mistakes.

The original download is in ../HiDF_runs/full-seed42/. Only the full variant was supplied. Ablation results require separate global and patch runs. The architecture uses fixed equal weights for the global and local logits.
