# 学習用コード

### 1. 学習環境作成

virtualenvを作成して依存パッケージをインストールします。

```
$ python3 -m venv .venv
$ . .venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install -r requirements.txt
```

### 2. コーパス準備

学習用のコーパスを準備します。  
公開された学習済みパッケージでは日本語版Wikipeidaのデータを[cirrussearchのdump data](https://dumps.wikimedia.org/other/cirrussearch/)からダウンロードして使用しました。

```
$ mkdir -p data/corpus
$ python make_split_corpus.py jawiki-20230424-cirrussearch-content.json.gz data/corpus
```

### 3. 設定ファイル作成

コーパス中の文字をカウント

```
$ python count_chars.py data/corpus
```

カウント結果を使用して設定ファイルを作成

```
$ python create_config.py
```

unidic_lite_imitatorでは、コーパス中の99.5%の文字に個別の文字IDを割り当て、登場頻度の少ない文字は未知語扱いにするよう設定ファイルを作成しています。

### 4. 学習データ作成

```
$ mkdir -p data/encoded
$ mkdir -p data/corpus_sampled
```

データの一部をサンプリングして評価・テストに使用
```
$ shuf data/corpus/corpus_31.txt | head -30000 > data/corpus_sampled/val.txt
$ shuf data/corpus/corpus_30.txt | head -30000 > data/corpus_sampled/test.txt
```

エンコード済み学習データを作成
```
$ python create_data.py configs/unidic_lite.json data/corpus_sampled/test.txt data/encoded/test.jsonl
$ python create_data.py configs/unidic_lite.json data/corpus_sampled/val.txt data/encoded/val.jsonl
$ python create_data.py configs/unidic_lite.json data/corpus/corpus_0.txt,data/corpus/corpus_1.txt,data/corpus/corpus_2.txt data/encoded/train.jsonl
```

### 5. 学習

```
$ python train.py configs/unidic_lite.json
```

### 6. 学習済みモデルを配布用の形式に変換

```
$ python to_onnx.py 192 PATH_TO_TRAINED_BEST_CHECKPOINT ../dist/python/unidic_lite_imitator/src/unidic_lite_imitator/model.onnx
$ python ckpt_to_pth.py PATH_TO_TRAINED_BEST_CHECKPOINT ../dist/python/unidic_lite_imitator_transformers/src/unidic_lite_imitator_transformers/model.pth
```
