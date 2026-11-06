# PyTorch + Hugging Face Fine-Tuning

A small, beginner-friendly project for fine-tuning a BERT-style model for sentiment classification. It uses **DistilBERT** (a compact BERT-family model), **PyTorch**, and the Hugging Face `datasets` and `transformers` libraries.

The training example uses the SST-2 split of the GLUE benchmark. You can change the model from the command line without editing the code.

## Quick start

### 1. Set up Python

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary by operating system and accelerator. If the command above does not install a suitable PyTorch build, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 2. Train

```bash
python train.py
```

The script downloads the model and SST-2 data on the first run, fine-tunes the model, evaluates it on the validation split, and saves the model and tokenizer to `outputs/distilbert-sst2/`.

To use the original uncased BERT model instead:

```bash
python train.py --model-name bert-base-uncased --output-dir outputs/bert-sst2
```

Training runs on CPU, but a compatible GPU is faster. If you run out of memory, lower `--batch-size` (for example, `--batch-size 8`).

### 3. Try a prediction

```bash
python predict.py --model-dir outputs/distilbert-sst2 --text "This tutorial is easy to follow."
```

The prediction output includes the sentiment label and confidence scores.

## Project files

- `train.py` — fine-tuning and validation in one readable script.
- `predict.py` — load a saved model and classify text.
- `requirements.txt` — Python dependencies.
- `tests/` — lightweight tests for the training utilities.

## Reproducibility and responsible use

Training and evaluation seeds are set for repeatability, but exact results can vary across hardware, software versions, and accelerator kernels. SST-2 is a small, English-language benchmark; its results do not guarantee performance on other text or populations. This example is for learning and experimentation, not high-stakes decisions.

This repository is licensed under the MIT License. The pretrained model and dataset have their own terms and usage conditions; check the model card and dataset documentation before using them. In particular, see the [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) and [GLUE dataset card](https://huggingface.co/datasets/nyu-mll/glue).

## Contributing and support

Bug reports and small improvements are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For security concerns, see [SECURITY.md](SECURITY.md).

## License

The project code is provided under the [MIT License](LICENSE).
