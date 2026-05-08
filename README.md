# Deep Learning SmolVLM-500M VQA Fine-Tuning Pipeline

This repository contains a state-of-the-art Vision-Language Model (VLM) fine-tuning pipeline built for multiple-choice visual question answering. It utilizes **Hugging Face's SmolVLM-500M-Instruct** architecture and implements advanced Kaggle-winning strategies, including dynamic resolution patching, Weight-Decomposed LoRA (DoRA), and Multilabel Stratified K-Fold cross-validation.

## Architecture & Methodology

* **Dynamic Pass-Through Sizing:** Hugging Face's SmolVLM-500M-Instruct defaults to scaling images up to a **2048x2048** ceiling (a 4x4 grid of 512px patches), which forces an average 445px image to explode from 64 visual tokens into 1,024 visual tokens. This pipeline bypasses that default by dynamically feeding the `current_longest_edge` to the processor. This preserves absolute high-res 760px clarity for dense diagrams while preventing a 16x token explosion on smaller images, massively reducing VRAM requirements.
* **Custom Dynamic Collator:** Safely zero-pads varying `pixel_values` and `pixel_attention_mask` tensors on the fly. This prevents `DataLoader` crashes by allowing variable-sized image patches (e.g., 4-patch high-res diagrams alongside 1-patch low-res photos) to coexist perfectly within the same training batch.
* **Attention-Max DoRA (The Winning Ablation):** Implements Weight-Decomposed LoRA targeting the core Attention routing network (`q_proj`, `k_proj`, `v_proj`, `o_proj`) at **Rank 16**. By decomposing weight updates for higher mathematical stability, increasing LoRA dropout to `0.10`, and utilizing a tuned learning rate of `4e-4` over 7 epochs, this configuration prevents overfitting while maximizing the VLM's spatial reasoning capabilities.
* **Ablation Studies:** Contains distinct modular pipelines for testing multiple architectures, including both **Baseline** (Direct A-J Alphabetical Extraction) and **Chain-of-Thought** (Rationale Generation) reasoning paths.

---

## Hardware Requirements

* **Target Hardware:** NVIDIA A100 GPU (40GB VRAM).
* **VRAM Usage:** The pipeline peaks at less than 20GB VRAM using a batch size of 4 with gradient accumulation of 4 (Effective Batch Size: 16).
* **Low-VRAM Environments:** For GPUs with less VRAM (Recommended 16GB), drop the `per_device_train_batch_size` to 2 and increase `gradient_accumulation_steps` to 8. An alternate 16 Batch Size (2x8) code block has been provided and commented out inside the training code of the ablation.

---

## Running Instructions

Remember to create a Virtual Environment and install the dependencies from `requirements.txt`.

### Windows Installation Commands

```cmd
py -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

```

---

## Repository Structure & Data Setup

Assuming you have the following relevant files and folders in your root directory, you can skip to the **Training & Inference** section (all ablations besides the final/winning one have been commented out):

* `combined_train_folds_full_multilabel.csv`
* `images\images\train_val_refined`
* `test_original.csv`

**Note on Directory Paths:** When it comes to adjusting the image paths within the `combined_train_folds` / `test` CSV files (due to having a different Base Directory), you can simply change the Base Directory variable inside of the `CONSTANTS` definition cell and uncomment the setup code towards the beginning of the Training & Inference section.

---

## Training & Inference

The codebase includes multiple historical ablation studies comparing different architectures and hyperparameters.

### Running the Winning Ablation

You can skip directly to the Inference section of the Final Ablation (**Ablation #5.3.4**) if the following adapter folder exists in your root directory:

* `smolvlm_fold_0_epoch_7_baseline_attention_max_lr4e-4_dora_higher_dropout`

*(If you do not have this folder locally, it can be downloaded from the provided project repository).*

Otherwise, you can acquire the other folders for the historical ablations by uncommenting the code inside the Data Analysis/Engineering section and/or the beginning of the Training & Inference section.

### Inference Output

The final inference script automatically matches the dynamic image patching used during training, extracts the predicted alphabetical answer (A-J), converts it back to standard integer indexing, and outputs a submission-ready CSV perfectly aligned for evaluation.