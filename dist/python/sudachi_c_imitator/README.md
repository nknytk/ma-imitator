# Sudachi C Imitator

This package imitates Japanese morphological analysis of Sudachi SplitMode.C and SudachiDict-full with a small onnx model.  
You can add tokenization and part-of-speech estimation to your environment  
with only 4MB additoinal disk space if you already have onnxruntime in your environment.

## Installation

```
$ pip install sudachi_c_imitator
```

## Usage Examples

```python
>>> import sudachi_c_imitator
>>> tagger = sudachi_c_imitator.Tagger()
>>> sample_text = '使い方のサンプルです。'
>>> tagger.parse(sample_text)
[('使い方', '名詞'), ('の', '助詞'), ('サンプル', '名詞'), ('です', '助動詞'), ('。', '補助記号')]
```

Input string length must be 256 or less.