# ma-imitator

辞書を使わず、機械学習により形態素解析の結果を再現することを目指すプロジェクトです。

## 学習済みパッケージ

### unidic_lite_imitator

日本語版Wikipediaのデータを使用し、mecab + unidic_liteの形態素解析を学習させたパッケージです。  
深層学習実行ランタイムを既にインストール済みの場合、unidic_lite_imitatorの追加インストールにより  
2MB程度の追加容量で日本語の単語分割と大分類までの品詞推定を行えるようになります。

* ONNX版
    - [pypiから](https://pypi.org/project/unidic-lite-imitator/)入手可能です。 `pip install unidic_lite_imitator`
    - [ONNX Runtime](https://pypi.org/project/onnxruntime/)を実行ランタイムに使用します。
    - 利用例
        ```python
        >> import unidic_lite_imitator
        >> tagger = unidic_lite_imitator.Tagger()
        >> sample_text = '使い方のサンプルです。'
        >> tagger.parse(sample_text)
        [('使い', '動詞'), ('方', '接尾辞'), ('の', '助詞'), ('サンプル', '名詞'), ('です', '助動詞'), ('。', '補助記号')]
        ```
* Hugging Face Transformers版
    - [pypiから](https://pypi.org/project/unidic-lite-imitator-transformers/)入手可能です。 `pip install unidic_lite_imitator_transformers`
    - [Transformers](https://pypi.org/project/transformers/), [PyTorch](https://pypi.org/project/torch/)を実行ランタイムに使用します。
    - 利用例
        ```python
        >> import unidic_lite_imitator_transformers
        >> tagger = unidic_lite_imitator_transformers.Tagger()
        >> sample_text = '使い方のサンプルです。'
        >> tagger.parse(sample_text)
        [('使い', '動詞'), ('方', '接尾辞'), ('の', '助詞'), ('サンプル', '名詞'), ('です', '助動詞'), ('。', '補助記号')]
        ```
* Web版
    - ONNX版のモデルをONNX Runtime Webを使いブラウザ上で動作させることが可能です。
    - 具体的な方法は[デモ](https://nknytk.github.io/presentations/demo/ma-imitator/unidic_lite_imitator.html)のソースコードを参照してください。

一度に入力できる最大文字数は192文字です。

テストデータにおけるmecab + unidic_liteとの一致度は下記の通りです。

| metrics | value |
| --- | --- |
| Precision | 0.974 |
| Recall | 0.971 |
| F1-Score | 0.973 |
| IOU | 0.947 |

なお、深層学習実行ランタイムは大きな容量を必要とします。深層学習を伴わない場面で本製品を利用することにメリットはありません。深層学習を伴わない場面で形態素解析が必要な場合、既存の辞書ベースの形態素解析機をご利用ください。

## 学習

`training/` 以下に学習用のコードが掲載されています。

## ディレクトリ構成

```
training/                                     学習に使うコード群
dist/                                         学習結果をパッケージ配布する際に使用するコード群
   |-- python/
      |-- unidic_lite_imitator/               ONNX版パッケージのソースコード
      |-- unidic_lite_imitator_transformers/  Transformers版パッケージのソースコード
```
