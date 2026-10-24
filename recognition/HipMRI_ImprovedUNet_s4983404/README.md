# HipMRI 2D Improved U-Net - s4983404

Research prototype for prostate/pelvic organ segmentation. Easy baseline: basic 2D U-Net;
Normal main model: a documented 2D adaptation of the approved Improved U-Net reference.
Hard is a conditional future option. Both 20-epoch models have measured frozen-test results below; anatomical label names remain unconfirmed.

## Working principles and reference mapping

Both networks encode images at five resolutions and concatenate encoder features into
the decoder. The baseline uses double 3x3 ReLU convolutions, max pooling and learned
transposed upsampling. Padded convolutions and width 16 are explicit practical
adaptations of [U-Net](https://arxiv.org/abs/1505.04597), not an exact 2015 reproduction.

The Normal architecture follows section 2.1 and figure 1 of
[Isensee et al. (2018)](https://arxiv.org/abs/1802.10508): residual context modules
(two pre-activation 3x3 convolutions with dropout 0.3), strided downsampling,
InstanceNorm, leaky ReLU slope 0.01, nearest upsampling followed by convolution,
3x3/1x1 localization with channel reduction, and three summed segmentation heads.
We replace 3D operators with 2D, four input modalities with one MRI channel, and
overlapping tumour outputs with six exclusive pelvic class logits. We use CE plus
foreground soft Dice, rather than claiming to reproduce the paper's entire training
recipe. Neither 2D network observes adjacent slices. Original code was written from
the architectural description; no student implementation was copied.

## Data and leakage prevention

Read-only root: `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`.
Preserve supplied train/validate/test folders: 11,460/660/540 slices, 31/4/3
filename-defined patients, no patient overlap. Repeated weeks stay in the same split.
No random slice split. `audit_data.py` rejects missing pairs, nonfinite pixels,
shape/affine mismatch and values outside integer IDs 0..5; it writes per-split manifests.
Normalization is per-image z-score, independent of all other images, with a constant
image guard. Preserve native shape without geometric resampling. Full audit found
11,400 training slices at 256x128 and 60 at 256x144; validation/test are 256x128.
Batch collation pads the bottom/right to the batch maximum; zero-sum target channels
mark padding, which is excluded from loss and confusion counts. Inference stays native.
Six one-hot channels remain present even on slices missing an organ. Optional training
augmentation flips the first array axis jointly; disabled for the first controlled runs.

Label IDs are currently reported numerically: 0..5 observed; 5 is an anatomical
prostate candidate supported by visual inspection but pending source confirmation. Do not treat that inference as
an authoritative label dictionary. Slice identity affines and 1-unit spacings do not
establish trustworthy original physical geometry; boundary distances should use pixels
until provenance is verified.

## Environment and use

Verified Rangpur versions: Python 3.11.13 (recorded in both completed runs),
PyTorch 2.14.0+cu130, NumPy 2.4.6, nibabel 5.4.2, matplotlib 3.11.1,
SciPy 1.17.1 (boundary analysis and its tests).
Use the existing `~/venv`. No nilearn or pytest needed. Local PyTorch cannot import
`c10.dll`; user approved equivalent CPU tests on Rangpur before GPU submission.

```bash
# Start inside recognition/HipMRI_ImprovedUNet_s4983404 on Rangpur.
source "$HOME/venv/bin/activate"
export OMP_NUM_THREADS=2
python -m unittest discover -s tests -v
python audit_data.py --output audit_new
# Small CPU inference check, using the original retained checkpoint:
python predict.py --device cpu --split validate --count 3 \
  --checkpoint "$HOME/hipmri_report_20261020_v4/runs/improved_full_679896/best.pt" \
  --output runs/improved_cpu_preview
```

The retained checkpoint path is in the student's account; another user must obtain
an authorized copy or retrain. Weights are not downloaded by cloning the repository.
Commands that write outputs require a new output directory. Do not overwrite previous results.
Use Slurm for GPU training, not the login node. No `--mem` request on this cluster;
`comp3710` jobs require `--account=comp3710`. `submit_smoke.sh` is 15 minutes on
`a100-test`. The measured 20-epoch runs use `sbatch submit_train.sh basic` or
`sbatch submit_train.sh improved` (20-minute budget and unique job-ID output folders).
Run JSON records config/versions/hardware; training saves validation-selected weights,
history, curves, log and resource measurements. Predict preserves native output affine.
Datasets and weights remain excluded from Git. Trained weight locations and
reproduction instructions are listed at the end of this report.
See [Reproduction guide](REPRODUCIBILITY.md) for a clean checkout, CPU checks,
training and evaluation commands, expected outputs, and verification evidence.
For new MRI slices without ground truth, `predict.py --input <file-or-directory>`
loads 2D NIfTI images directly and generates input/prediction panels. Optional
`--labels <directory>` adds ground-truth comparisons; see the guide for naming.

## Evaluation conventions

Exclusive argmax predictions, no tuned threshold. Aggregate integer confusion counts
over pixels for per-class Dice=2TP/(2TP+FP+FN), IoU=TP/(TP+FP+FN).
Both-empty class is undefined (`null`), not a perfect score; one-empty gives zero.
Also report patient-level pooled counts; test has only three patients.
Checkpoints use validation label-5 pooled Dice provisionally; confirm its anatomical
meaning before interpreting this as a confirmed prostate score. No test feedback enters model selection.
GPU latency: batch one, five warmups, 30 synchronized forwards, excluding I/O;
peak allocated and reserved memory includes training. This is slice latency, not
full-volume latency. Validation failure analysis includes pixel HD95; millimetre distances require trustworthy geometry.

## Frozen 20-epoch experiment results

Numeric class IDs are retained: anatomical label mapping is still unconfirmed.
Checkpoints were selected using validation only; test argmax was evaluated once per model.

| Model / job | Epochs / selected epoch | Best val ID5 Dice | Test ID5 Dice | Test ID5 IoU |
|---|---:|---:|---:|---:|
| basic / 679897 | 20 / 19 | 0.8258 | 0.8453 | 0.7321 |
| improved / 679896 | 20 / 9 | 0.8466 | 0.8614 | 0.7566 |

| ID | Basic Dice | Improved Dice | Basic IoU | Improved IoU |
|---|---:|---:|---:|---:|
| 0 | 0.9965 | 0.9968 | 0.9931 | 0.9936 |
| 1 | 0.9838 | 0.9850 | 0.9682 | 0.9704 |
| 2 | 0.9344 | 0.9376 | 0.8768 | 0.8826 |
| 3 | 0.9652 | 0.9646 | 0.9328 | 0.9317 |
| 4 | 0.8314 | 0.8635 | 0.7115 | 0.7599 |
| 5 | 0.8453 | 0.8614 | 0.7321 | 0.7566 |

| Patient | Basic ID5 Dice | Improved ID5 Dice |
|---|---:|---:|
| 040 | 0.8521 | 0.8662 |
| 041 | 0.8295 | 0.8259 |
| 042 | 0.8341 | 0.8664 |

| Model | Parameters | Peak allocated / reserved MiB | Train loop s | Batch-one latency ms |
|---|---:|---:|---:|---:|
| basic | 1,940,902 | 661.6 / 974.0 | 706.0 | 2.06 |
| improved | 3,189,714 | 2026.6 / 3026.0 | 916.7 | 5.84 |

Training-loop time excludes cache loading. Slurm wall time is recorded in the retained logs.
Latency excludes I/O; five warmups, 30 synchronized batch-one forwards on A100-PCIE-40GB.
Both models use the frozen shared recipe in [evaluation plan](EVALUATION_PLAN.md).

![Basic learning curves](figures/basic_learning_curves.png)

![Improved learning curves](figures/improved_learning_curves.png)

## Validation boundary failures

Basic: five worst positive-label slices. Improved: two worst misses plus three explicit error modes.
Selection is deliberately biased toward failures; these panels are not test averages.
ID5 remains an anatomical candidate. HD95 uses symmetric pooled 4-connected surface distances in pixels.
Both-empty and one-empty distances are undefined, with reasons retained in JSON.

**basic:** median finite slice HD95 4.00 px (237 finite; 389 both-empty; 34 one-empty). Finite-only median omits total misses.

| Slice | Dice | FP / FN pixels | HD95 px | Predicted components |
|---|---:|---:|---:|---:|
| case_036_week_0_slice_17.nii.gz | 0.0000 | 0 / 17 | undefined | 0 |
| case_037_week_0_slice_10.nii.gz | 0.0000 | 0 / 56 | undefined | 0 |
| case_037_week_0_slice_11.nii.gz | 0.0000 | 0 / 84 | undefined | 0 |
| case_037_week_0_slice_50.nii.gz | 0.0000 | 0 / 155 | undefined | 0 |
| case_037_week_0_slice_51.nii.gz | 0.0000 | 0 / 35 | undefined | 0 |

![basic failure 1: input, GT, prediction, red FP and blue FN](figures/basic_failure_1.png)

![basic failure 2: input, GT, prediction, red FP and blue FN](figures/basic_failure_2.png)

![basic failure 3: input, GT, prediction, red FP and blue FN](figures/basic_failure_3.png)

![basic failure 4: input, GT, prediction, red FP and blue FN](figures/basic_failure_4.png)

![basic failure 5: input, GT, prediction, red FP and blue FN](figures/basic_failure_5.png)

**improved:** median finite slice HD95 3.61 px (240 finite; 393 both-empty; 27 one-empty). Finite-only median omits total misses.

| Slice | Dice | FP / FN pixels | HD95 px | Predicted components |
|---|---:|---:|---:|---:|
| case_036_week_0_slice_45.nii.gz | 0.0000 | 0 / 64 | undefined | 0 |
| case_037_week_0_slice_49.nii.gz | 0.0000 | 0 / 315 | undefined | 0 |
| case_038_week_0_slice_35.nii.gz | 0.6823 | 231 / 0 | 5.00 | 1 |
| case_038_week_0_slice_27.nii.gz | 0.6057 | 16 / 53 | 4.00 | 2 |
| case_038_week_0_slice_42.nii.gz | 0.0000 | 15 / 56 | 23.59 | 1 |

![improved failure 1: input, GT, prediction, red FP and blue FN](figures/improved_failure_1.png)

![improved failure 2: input, GT, prediction, red FP and blue FN](figures/improved_failure_2.png)

![improved failure 3: input, GT, prediction, red FP and blue FN](figures/improved_failure_3.png)

![improved failure 4: input, GT, prediction, red FP and blue FN](figures/improved_failure_4.png)

![improved failure 5: input, GT, prediction, red FP and blue FN](figures/improved_failure_5.png)


## Failure autopsy and engineering recommendation

The five Improved examples above were visually inspected. Counts are observations;
appearance-based explanations below are hypotheses, not confirmed anatomical causes.
The last three panels were rendered on CPU from the same saved checkpoint. Aggregate
test tables retain the original frozen GPU pass. Corrected validation analysis matches
checkpoint Dice within 0.00002; batch-one versus batch-16 floating-point kernels can
change a few boundary pixels. The earlier failed image export was excluded.

| Actual validation case | Paired baseline / Improved ID5 Dice | Observed error and likely trigger | Practical response |
|---|---:|---|---|
| 036, week 0, slice 45 | 0.4976 / 0.0000 | Improved omits all 64 target pixels while baseline retains 52. A small local region is suppressed among surrounding tissue; added complexity does not consistently preserve small targets. | Review empty predictions near mask transitions; do not infer safety from overall Dice. |
| 037, week 0, slice 49 | 0.1120 / 0.0000 | All 315 target pixels are missed by Improved. In slices 49-52 the target shrinks from 315 to 155, 35 and 5 pixels, and all are omitted. This is a transition failure, not merely a single one-pixel error. | Inspect neighbouring slices together. Call this a mask-defined transition until the original plane and anatomical direction are established. |
| 038, week 0, slice 35 | 0.7778 / 0.6823 | Improved includes all 248 target pixels but adds 231 FP pixels, versus 137 FP for baseline. The red ring shows boundary expansion around a poorly separated region. | Require local contour review; improved global Dice does not guarantee better local specificity. |
| 038, week 0, slice 27 | 0.0000 / 0.6057 | Improved recovers 53 target pixels, misses 53 and adds 16 FP pixels; its prediction has two 4-connected components. The image shows a split small region. | Treat disconnected target proposals as review candidates. Any component filtering must be validated rather than assumed harmless. |
| 038, week 0, slice 42 | 0.0000 / 0.0000 | Improved misses all 56 target pixels and predicts a separate 15-pixel island. HD95 is 23.59 pixels. Baseline misses the target without this extra island. | A plausible-looking isolated contour can be misplaced; a single-component rule alone would not detect this failure. |

Across all 660 validation slices, complete positive-target misses fall from 17 to 14
and false-positive-only slices from 17 to 13. Median finite HD95 falls from 4.00 to
3.61 pixels, but finite-only summaries exclude one-empty failures and use slightly
different slice populations. These are descriptive validation findings, not proof of
clinical boundary benefit or millimetre accuracy.

On the frozen test split, Improved increases pooled ID5 Dice by 0.0161 and IoU by
0.0245. It improves two of three patients, while patient 041 decreases from 0.8295 to
0.8259. Peak allocated VRAM is about 3.06 times baseline, forward latency 2.84 times,
and training-loop time 1.30 times. Only three test patients and one training seed
support these observations; slice-wise significance claims would be unjustified.

**Recommendation:** retain Improved as the coursework Normal model and the basic
U-Net as the implemented comparison/fallback. Both fit comfortably in the measured
resource budget. Use the prototype only for expert-reviewed drafts; verify the ID5
label dictionary and original geometry before asserting the prostate target or
anatomical base/apex findings. The modest pooled gain and persisting regressions do
not justify claiming that 2D architectural changes solve boundary safety. Defer Hard
3D work until these source questions and additional time/data requirements are clear.

## Midpoint feasibility review

Student: s4983404 | Draft for tutor discussion

**1. Aim and success criteria**

I aim to produce prostate outlines that an imaging specialist can review and correct. I will compare a basic U-Net with an Improved U-Net to see whether the extra complexity gives better boundaries at a reasonable computing cost.

- Keep all images from the same patient in one data split.
- Target prostate test Dice >= 0.75 and report Dice and IoU for each label.
- Compare the models and inspect 3-5 examples of boundary errors.
- Keep GPU memory below 16 GB and record training and prediction time.

**2. Model and data**

I will use HipMRI 2D slices and their segmentation masks. A basic 2D U-Net (Easy) is the baseline. A 2D Improved U-Net (Normal), based on Isensee et al. (2018), is the main model. This uses course ideas including convolution, skip connections, normalization and validation. Both models will use the same data split and preprocessing.

**3. Preliminary evidence**

The image and mask pairs can be loaded together. The supplied split separates 31 training, 4 validation and 3 test patients. A short training trial has run successfully on Rangpur. These checks support feasibility, but do not establish reliability on new patients.

**4. Risks and next steps**

The test group is small, and some boundaries are unclear. I need to confirm the label numbers and slice direction. Next, I will review the model comparison, local errors and computing cost. Runs are budgeted at up to 20 minutes each. The basic U-Net is the fallback if the Improved model gives little benefit. 3D work is optional if time allows.

References: Ronneberger et al. (2015), U-Net; Isensee et al. (2018), BRATS 2017 contribution. Data: HipMRI Study.

The authoritative one-page review is [HipMRI_Feasibility_Checkoff.pdf](HipMRI_Feasibility_Checkoff.pdf), copied unchanged from the student's workspace-root version. The student confirmed completion of the midpoint tutor check-off on 24 October 2026. The original PDF retains its pre-consultation draft/pending footer; that footer does not describe the current completion status.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with requirement extraction, architecture research, data-audit
scripts, original model/training code, tests and feasibility drafting. Verification
uses strict CPU tests, exhaustive pair/pixel auditing, real-data overfit, scheduled
GPU trials and checkpoint reload. These checks and experiments were completed; label provenance remains unresolved. The
student must review, understand and defend every implementation and reported claim.

## Data acknowledgment

Data were provided by HipMRI Study via the COMP3710 course dataset. Follow its
noncommercial data-use agreement and do not identify or contact participants.
The agreement recommends https://doi.org/10.1016/j.ijrobp.2015.08.045 for MRI-alone
treatment planning; its segmentation citation field is blank. No invented segmentation
reference is supplied. Exact dataset provenance and anatomical label dictionary remain
to be confirmed.

## Retained weights and reproducibility

Weights remain outside Git on Rangpur:

- `~/hipmri_report_20261020_v4/runs/basic_full_679897/best.pt`
- `~/hipmri_report_20261020_v4/runs/improved_full_679896/best.pt`

Use `predict.py --checkpoint <path> --output <new-directory>` in the existing venv. A public weight link has not been published. Course users with dataset access can retrain using the documented commands. All plots and numeric summary files in this folder are derived from the identified runs; raw datasets and model files are excluded.

Audit guarantees are limited to filename-defined subject separation and numeric grid/label checks; they do not establish original provenance, absence of hidden duplicates or clinical annotation correctness. The submission branch is `topic-recognition` in the student's personal fork, targeting upstream `shakes76/PatternAnalysis-2026:topic-recognition`.
