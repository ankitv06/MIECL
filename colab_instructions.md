# Running MIECL Echo-Chamber Debiasing on Google Colab (T4 GPU)

This guide provides step-by-step instructions to set up and run the model on Google Colab using a T4 GPU.

---

## 1. Colab Notebook Setup

Create a new notebook in Google Colab and set the runtime to **T4 GPU** (Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU).

### Step 1: Mount Google Drive
Mount your Google Drive to access the dataset and save the trained models/predictions.
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Prepare Code Directory
Copy or clone the code files into `/content/MIECL` and switch to that directory:
```bash
# Assuming code is in your Drive or you want to copy it to local runtime
!mkdir -p /content/MIECL
!cp -r "/content/drive/MyDrive/News Recc Code/"* /content/MIECL/
%cd /content/MIECL
```

### Step 3: Install/Download Dependencies
MIND dataset parsing uses the `nltk` word tokenizer, which requires the `punkt` resource:
```python
import nltk
nltk.download('punkt')
```

---

## 2. Running Training and Evaluation

Execute the following bash command in Colab to run training and evaluation with the new **echo-chamber debiased** mode:

```bash
!python main.py \
  --infonce_mode echo_chamber_debiased \
  --contrastive_mode USER \
  --dataset_dir "/content/drive/MyDrive/News Recc Code/dataset" \
  --glove_path "/content/drive/MyDrive/News Recc Code/dataset/glove.840B.300d.txt" \
  --preserve_dir "/content/drive/MyDrive/News Recc Code/outputs/echo-chamber-debiased" \
  --num_epoch 6 \
  --batch_size 30 \
  --alpha 1.0
```

### Parameters explained:
* `--infonce_mode echo_chamber_debiased`: Activates the newly implemented echo-chamber hard negative sampling.
* `--contrastive_mode USER`: Enables contrastive learning loss.
* `--dataset_dir`: Pointing to your Google Drive dataset location.
* `--glove_path`: Path to Glove embeddings.
* `--preserve_dir`: Output directory to save prediction files, evaluation scores, and PyTorch model checkpoints.

---

## 3. Compatibility Verified
* **Single GPU (T4) Support**: The codebase has been updated to dynamically allocate the model and tensors to the GPU (`cuda`) if available, avoiding device mismatch errors.
* **Auto-Fallback to CPU**: If run locally or on a standard CPU runtime, the code automatically falls back to CPU execution without modification.
* **No hardcoded Multi-GPU IDs**: Multi-GPU wrapping (`nn.DataParallel`) is now dynamically applied only if multiple GPUs are available, preventing execution errors on a single Colab GPU.
