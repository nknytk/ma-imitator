# ma-imitator

辞書を使わず、機械学習により形態素解析の結果を再現することを目指すプロジェクトです。

## 学習済みパッケージ

日本語版Wikipediaのデータを使用し、形態素解析器の結果を学習させたモデルがPyPIで公開されています。  
学習させた形態素解析器と辞書の組み合わせは3パターンあり、それぞれに対して実行ランタイムがONNX RuntimeのものとHugging Face Transformers on PyTorchのものの2パッケージが用意されています。深層学習実行ランタイムを既にインストール済みの場合、これらのパッケージを利用することで、日本語の単語分割と大分類までの品詞推定を数MBの追加容量で行えるようになります。

1. mecab + unidic_lite
    * ONNX版: [unidic_lite_imitator](https://pypi.org/project/unidic-lite-imitator/)
    * Transformers版: [unidic_lite_imitator_transformers](https://pypi.org/project/unidic-lite-imitator/)
2. Sudachi SplitMode.B + SudachiDict-full
    * ONNX版: [sudachi_b_imitator](https://pypi.org/project/sudachi-b-imitator/)
    * Transformers版: [sudachi_b_imitator_transformers](https://pypi.org/project/sudachi-b-imitator/)
3. Sudachi SplitMode.C + SudachiDict-full
    * ONNX版: [sudachi_c_imitator](https://pypi.org/project/sudachi-c-imitator/)
    * Transformers版: [sudachi_c_imitator_transformers](https://pypi.org/project/sudachi-c-imitator/)

### インストール方法

パッケージ名を指定してpip installしてください。

```
$ pip install ${PACKAGE_NAME}
```

### 利用方法

各パッケージのTaggerインスタンスを作成し、`parse()`メソッドの引数に文字列を与えてください。なお、一度に入力できる最大文字数は256文字です。
利用方法は上記全てのパッケージで共通です。

利用例:

```python
from unidic_lite_imitator import Tagger
tagger = Tagger()
sample_text = '使い方のサンプルです。'
print(tagger.parse(sample_text))
# 下記の内容が標準出力されます
# [('使い', '動詞'), ('方', '接尾辞'), ('の', '助詞'), ('サンプル', '名詞'), ('です', '助動詞'), ('。', '補助記号')]
```

### パッケージごとの出力の差異

単語を短く分割する順にunidic_lite, Sudachi B, Sudachi Cとなります。

例:

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

### 精度

各モデルのテストデータにおける、辞書を使用した形態素解析器との一致度は下記の通りです。

| metrics | unidic_lite | Sudachi SplitMode B | Sudachi SplitMode C |
| --- | --- | --- | --- |
| Precision | 0.975 | 0.963 | 0.964 |
| Recall | 0.971 | 0.962 | 0.964 |
| F1-Score | 0.973 | 0.962 | 0.964 |
| IOU | 0.948 | 0.927 | 0.931 |

なお、深層学習実行ランタイムは大きな容量を必要とします。深層学習を伴わない場面で本製品を利用することにメリットはありません。深層学習を伴わない場面で形態素解析が必要な場合、既存の辞書ベースの形態素解析機をご利用ください。

## 学習

`training/` 以下に学習用のコードが掲載されています。
