# Local evidence and JPEG consistency for HiDF face detection

## Train in Colab

Create `MyDrive/HiDF/` in Google Drive. Upload the existing `Fake-img/`, `Real-img/`, and `hidf_clean_source_manifest.csv` along with this entire `patch_detector/` folder. Open `train_colab.ipynb` in Google Colab, select a GPU runtime, and run the setup and training cells. The notebook stages images on the runtime disk and saves each completed epoch to Drive. Rerun those cells after a disconnect. The final cell evaluates the best checkpoint and exports individual predictions.

## Research question

Does adding a local evidence head improve held-out face classification, and does penalizing disagreement between two JPEG views preserve fake recall after recompression? TriLens's saved fake recall falls from 0.9784 on clean images to 0.8314 at JPEG quality 50. Use that observation to motivate the experiment and disclose that the existing test results informed the question. An independent dataset or a fresh locked holdout would strengthen a subsequent confirmation.

The model scores an RGB image with ResNet-18. A second head scores each location on the intermediate 16 × 16 feature map. It averages the largest 32 local scores, then averages that result with the whole-image logit. Training uses matched original and recompressed views with a shared horizontal flip. Both views receive the image label. The full objective adds 0.5 times the mean squared difference between the two predicted fake probabilities. Evaluation requires one forward pass per image per condition.

Patch-based forgery detection has established prior work. Cite that work, the backbone and HiDF, and describe this study as testing a particular combination and training objective. A broader literature review and independent experiments are needed to establish novelty and generalization. [Chai et al., ECCV 2020](https://chail.github.io/patch-forensics/) · [Torchvision ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) · [HiDF](https://doi.org/10.1145/3711896.3737399)

## Controlled experiments

| Variant | Global head | Local head | Paired JPEG training | Consistency penalty |
| --- | --- | --- | --- | --- |
| global | Yes | No | Yes | No |
| patch | Yes | Yes | Yes | No |
| full | Yes | Yes | Yes | Yes |

Select all three variants and seeds 42, 43 and 44 in the training cell for the paper experiment. Use identical batches, epoch budgets and pretrained initialization across these variants. Select checkpoints using validation ROC-AUC. Freeze settings before the final test. All variants use 256 × 256 input and classify an image as fake when its fake probability exceeds 0.5.

For the team comparison, use identical split assignments, fake-positive labels, decision rule and JPEG pipeline on all three members' models. Report differences in pretraining, optimizer, augmentation and training budget because each can affect the ranking. Record mean and standard deviation across seeds, clean ROC-AUC, clean and Q50 fake recall, and real-image false-positive rate. Show absolute clean and compressed scores alongside the drop. If comparing uncertainty on the same test images, use paired resampling of entire source groups.

Each run saves settings and library versions in `run.json`, validation results in `history.json`, and the best and most recent checkpoints. Final evaluation produces `test.json` and one CSV per condition with image ID, label, source group and fake probability. These files support comparison of the same mistakes across team members. Preserve the notebook settings and code alongside the exported results.
