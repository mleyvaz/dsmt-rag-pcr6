# Data sources and integrity

## RAMDocs

- Official repository: https://github.com/HanNight/RAMDocs
- File: `RAMDocs_test.jsonl`
- Retrieved: 2026-09-14
- Records: 500
- SHA-256: `C67F699C97349F00CF1BD08D1DBF8CA1D0CC38C306715C93A10A4F961DCF28B7`
- Repository license: MIT

## ConflictQA

- Official dataset: https://huggingface.co/datasets/osunlp/ConflictQA
- Configuration file: `conflictQA-popQA-chatgpt.json`
- Retrieved: 2026-09-15
- Records: 7,947
- SHA-256: `835F7D80D009D10B077551779C0DECFAE6EDE4EF7CFCFDC0C5148EDA30516A2F`
- Dataset license: Apache 2.0

The raw datasets remain in the private work directory and are not duplicated in the output package. The output package contains derived metrics and item-level predictions required to audit the analyses.

## NLI model

- Model: `Xenova/distilbert-base-uncased-mnli`
- Model page: https://huggingface.co/Xenova/distilbert-base-uncased-mnli
- Runtime: Transformers.js 4.2.0 with ONNX Runtime
- Quantization: 8 bit
- Maximum sequence length: 256 tokens
- RAMDocs document–candidate pairs: 7,802
- Label order verified from model configuration: entailment, neutral, contradiction
