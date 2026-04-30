# Running OpenAI's Whisper on an NPU
## DATA 255, Group 1 - Abhi, Akhilesh, Daniel, and Shriansh

This is our codebase for training an OpenAI Whisper Tiny model and running it on a Mobilint Neural Processing Unit (NPU). In addition, we tested whether adding a couple of changes would find improvements in model latency or model speed. These changing the positional embedding of the attention mechanism from sinusoidal positional embeddings to rotary positional embedding (RoPE) and modifying the attention mechanism to use FlashAttentionV3.

We trained our models with the [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) and [Common Voice](https://mozilladatacollective.com/datasets/cmndapwry02jnmh07dyo46mot) datasets.

After training our baseline (unmodified) and experimental (RoPE + FlashAttention) variants of Whisper, we would compile these models to MXQ files using `qubee` to run them on Mobilint NPUs using `maccel`.

### How to prepare:

1. Python venv setup

```bash
# Create virtual environment
python -m venv .venv

# Activate the environment
.\.venv\Scripts\activate # Windows

source .venv/bin/activate # Unix

# Install packages
pip install -r requirements.txt

# Recommended to install torch after
pip3 install torch --index-url https://download.pytorch.org/whl/cu130
```

2. Create `.env` file

Required for downloading Common Voice dataset and optional for downloading LibriSpeech dataset.

```env
MDC_API_KEY="your api key" # Required
HF_TOKEN="your api key" # Optional
```

3. Download and extract Common Voice dataset
```bash
python download_commonvoice.py # Takes a bit of time to download an 87 GB tarball

python extract_cv.py
# OR
tar -xkzvf common-voice-scripted-speech-25-0-englis-0c0b9a16.tar.gz -C cv-data
# The tar command is probably faster than the Python script, but both will take over a day to complete.
```

4. Begin training

If you wish to train the experimental odel, be sure to change line 4 from
```python
from my_model_config import get_model
```

to 
```python
from my_model_config_rope import get_model
```

```bash
python train.py
```

Because we train this model for 1 million epochs, this training process will take a long time. If you pause it in the middle, be sure to uncomment line 49 of `train.py` and put in the name of the last checkpoint before running again.

5. Conversion to ONNX

On line 29 of `export_onnx.py`, alter the path of the checkpoint to match that of the checkpoint to be altered.

```bash
python export_onnx.py
```

6. Compilation to MXQ

This step requires you to have a wheel for `qubee`. Our wheel is version 0.12.0.

    1. Set up a docker container for qubee compilation

```bash 
docker run -it --name qb_compiler_container ^
  -v /your/path/here:/work ^
  mobilint/qbcompiler:0.12-cpu-ubuntu22.04 /bin/bash
```

    2. Move the qubee wheel into the mounted directory and then install it.

```bash
python -m pip install qubee-0.12.0.0+aries2-py3-none-any.whl
```

    3. Install other libraries necessary for the compilation process.

```bash
python -m pip install librosa
python -m pip install torch torchvision # apparently this downloads the latest CUDA release
```

    4. Copy ONNX files and compilation scripts

```bash
cp whisper_encoder.onnx /your/path/here
cp whisper_decoder.onnx /your/path/here
cp onnx_to_mxq.py /your/path/here
```

    5. Create and copy over a cv-test directory

TBD

    6. Compile the models
```bash
python onnx_to_mxq.py
```

7. Running on NPU

Requires a machine with a Mobilint Aries NPU set up and a wheel of `maccel`.

    1. Install the `maccel` wheel.

```bash
python -m pip install maccel-0.30.1-cp312-cp312-win_amd64.whl
```

    2. Install other necessary libraries

```bash
pip install librosa jiwer numpy torch tiktoken
```

    3. Run on the NPU

```bash
python npu_infer.py
```
