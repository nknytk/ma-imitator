# ma-imitator

This project aims to reproduce output of Japanese morphological analyzers by machine learning without dictionaries.

## Trained Packages

You can get trained models from PyPI. These models were trained on Japanese Wikipedia data to reproduce output of popular Japanese morphological analyzers.
3 patterns of combination of morphological analyzers and dictionaries are used as teachers. An ONNX Runtime backend package and a Hugging Face Transgormers on PyTorch backend package are available for each pattern. If you already have deep learning runtimes on your environment, you can add Japanese word segmentation and part-of-speech estimation with only a few MB of additional disk space.

1. mecab + unidic_lite
    * ONNX version: [unidic_lite_imitator](https://pypi.org/project/unidic-lite-imitator/)
    * Transformers version: [unidic_lite_imitator_transformers](https://pypi.org/project/unidic-lite-imitator/)
2. Sudachi SplitMode.B + SudachiDict-full
    * ONNX version: [sudachi_b_imitator](https://pypi.org/project/sudachi-b-imitator/)
    * Transformers version: [sudachi_b_imitator_transformers](https://pypi.org/project/sudachi-b-imitator/)
3. Sudachi SplitMode.C + SudachiDict-full
    * ONNX版: [sudachi_c_imitator](https://pypi.org/project/sudachi-c-imitator/)
    * Transformers版: [sudachi_c_imitator_transformers](https://pypi.org/project/sudachi-c-imitator/)

Note: Deep learning runtimes requires large disk space. There is no advantage in using the above packages in situations that do not involve deep learning. Using exisiting dictionary-based morphological analyzers is strongly recommended for tasks that do not involve deep learning.

### Installation

You can install these packages with pip.

```
$ pip install ${PACKAGE_NAME}
```

### Usage

Initialize Tagger instance from each package and give a string as an argument to the `parse()` method. The maximum length of the input string is 256 characters.  
This user interface is same for all the above packages.

Usage example:

```python
from unidic_lite_imitator import Tagger
tagger = Tagger()
sample_text = '使い方のサンプルです。'
print(tagger.parse(sample_text))
# 下記の内容が標準出力されます
# [('使い', '動詞'), ('方', '接尾辞'), ('の', '助詞'), ('サンプル', '名詞'), ('です', '助動詞'), ('。', '補助記号')]
```

If your input string is longer than 256 characters, download corresponding `config.json` and `model.onnx` or `model.pth` from [dist/long](./dist/long) and replace from original ones. Then you can input a string of up to 1024 characters in exchange for a decrease in processing speed.

You can [try ONNX version models on browser](https://nknytk.github.io/presentations/demo/ma-imitator/unidic_lite_imitator.html) with ONNX Runtime Web.

### Difference of the Models

The order in which the words are divided into shorter segments is: unidic_lite, Sudachi B, Sudachi C.

Example:

```python
>>> import unidic_lite_imitator
>>> import sudachi_b_imitator
>>> import sudachi_c_imitator
>>> sample_text = '安倍前総理大臣が成田空港に到着した。'
>>> unidic_lite_imitator.Tagger().parse(sample_text)
[('安倍', '名詞'), ('前', '名詞'), ('総理', '名詞'), ('大臣', '名詞'), ('が', '助詞'), ('成田', '名詞'), ('空港', '名詞'), ('に', '助詞'), ('到着', '名詞'), ('し', '動詞'), ('た', '助動詞'), ('。', '補助記号')]
>>> sudachi_b_imitator.Tagger().parse(sample_text)
[('安倍', '名詞'), ('前', '名詞'), ('総理大臣', '名詞'), ('が', '助詞'), ('成田空港', '名詞'), ('に', '助詞'), ('到着', '名詞'), ('し', '動詞'), ('た', '助動詞'), ('。', '補助記号')]
>>> sudachi_c_imitator.Tagger().parse(sample_text)
[('安倍', '名詞'), ('前総理大臣', '名詞'), ('が', '助詞'), ('成田空港', '名詞'), ('に', '助詞'), ('到着', '名詞'), ('し', '動詞'), ('た', '助動詞'), ('。', '補助記号')]
```

### Performance

Scores of the above models on test data are shown below.

| metrics | unidic_lite | Sudachi SplitMode B | Sudachi SplitMode C |
| --- | --- | --- | --- |
| Precision | 0.975 | 0.963 | 0.964 |
| Recall | 0.971 | 0.962 | 0.964 |
| F1-Score | 0.973 | 0.962 | 0.964 |
| IOU | 0.948 | 0.927 | 0.931 |

## Training

`training/` directory holds codes for training.

## Acknowledgments

Published packages are trained with Cloud TPUs provided by [TPU Research Cloud](https://sites.research.google/trc/about/) program.
