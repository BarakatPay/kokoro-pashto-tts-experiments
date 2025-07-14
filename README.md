# Kokoro Pashto TTS Experiments

This repository contains experiments training a Transformer-based TTS model on the Pashto **Mangal Paktika** dataset using the `SimpleTTSModel` class.

---

## Dataset

- **Name**: Mangal Paktika Pashto speech corpus
- **Location**: `/content/drive/MyDrive/Mangal_Paktika_Audios`
- **Metadata file**: `mangal_paktika_entries.json`
- **Number of samples**: `22814`
- **Number of batches**: `11407`

---

## Configurations

### Paths

- **Google Drive mount root**:
  ```python
  DRIVE_PATH = "/content/drive/MyDrive"
  ```
- **Audio folder**:
  ```python
  AUDIO_FOLDER = f"{DRIVE_PATH}/Mangal_Paktika_Audios"
  ```
- **Metadata JSON**:
  ```python
  JSON_FILE = f"{DRIVE_PATH}/mangal_paktika_entries.json"
  ```
- **Output directory** (checkpoints & logs):
  ```python
  OUTPUT_DIR = f"{DRIVE_PATH}/kokoro_pashto_models"
  ```

---

### Model

- **Base weights**: `"hexgrad/Kokoro-82M"` (via HuggingFace)
- **Model class**: `SimpleTTSModel`

---

### Preprocessing

- **Sample rate**: `22050`
- **Mel spectrogram parameters**:
  - `n_mels`: `80`
  - `n_fft`: `1024`
  - `hop_length`: `256`
  - `win_length`: `1024`
- **Text/sequence length limits**:
  - `MAX_TEXT_LENGTH` (max words): `100`
  - `TARGET_MEL_LENGTH` (frames): `800`
  - `MIN_MEL_LENGTH` (frames): `50`

---

### Architecture

- **Hidden dimension** (`D_MODEL`): `512`
- **Attention heads** (`N_HEADS`): `8`
- **Transformer layers** (`N_LAYERS`): `6`
- **Upsample factor**: `8`

---

### Training

- **Epochs**: `10`
- **Batch size**: `2`
- **Learning rate**: `1e-4`
- **Gradient accumulation steps**: `4`
- **Max grad norm**: `1.0`
- **Optimizer**: `AdamW`
- **Loss**: `MSELoss`
- **Device**: `cuda` (if available) or `cpu`

Training progress is displayed via `tqdm`, with checkpoints saved every 500 steps under `OUTPUT_DIR`.

---

## Files & Structure

```
.
├── Kokoro_Pashto_TTS.ipynb       # Colab notebook
├── checkpoints/                  # model checkpoints
│   └── epoch_{n}.pth
└── README.md                     # this file
```

---

## Usage

1. **Clone the repo**
   ```bash
   git clone https://github.com/BarakatPay/kokoro-pashto-tts-experiments.git
   ```
2. **Open** `Kokoro_Pashto_TTS.ipynb` in Colab.
3. **Mount** your Google Drive so that `/content/drive/MyDrive/...` paths resolve.
4. **Install** dependencies:
   ```bash
   pip install torch torchaudio librosa soundfile transformers tqdm
   ```
5. **Run** all cells to start training.
6. **Check** `checkpoints/` (under `OUTPUT_DIR`) for saved models.

> **Note:** Training is in progress; loss curves and audio samples will be documented here once complete.

