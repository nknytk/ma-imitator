import json
import sys

configs = {
  "unidic_lite": {
    "description": "mecab + unidic-liteを模した単語分割・品詞推定を行うための設定ファイル 256文字",
    "parts_of_speech": [
      "名詞", "代名詞", "形状詞", "連体詞", "副詞", "接続詞", "感動詞", "動詞", "形容詞",
      "助動詞", "助詞", "接頭辞", "接尾辞", "記号", "補助記号", "空白"
    ],
    "attention_probs_dropout_prob": 0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0,
    "embedding_size": 48,
    "hidden_size": 116,
    "initializer_range": 0.02,
    "intermediate_size": 468,
    "max_position_embeddings": 256,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "char_coverage": 0.995,
    "chars": ""
  },
  "sudachi": {
    "description": "sudachiを模した単語分割・品詞推定を行うための設定ファイル 256文字",
    "parts_of_speech": [
      "名詞", "代名詞", "形状詞", "連体詞", "副詞", "接続詞", "感動詞", "動詞", "形容詞",
      "助動詞", "助詞", "接頭辞", "接尾辞", "記号", "補助記号", "空白"
    ],
    "attention_probs_dropout_prob": 0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0,
    "embedding_size": 64,
    "hidden_size": 156,
    "initializer_range": 0.02,
    "intermediate_size": 624,
    "max_position_embeddings": 256,
    "num_attention_heads": 6,
    "num_hidden_layers": 4,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "char_coverage": 0.999,
    "chars": ""
  }
}
with open('data/chars.csv') as fp:
    for row in fp:
        rank, _rest = row.strip().split(',', 1)
        c, count, cummulative_coverage = _rest.rsplit(',', 2)
        for config in configs.values():
            if float(cummulative_coverage) <= config['char_coverage']:
                config['chars'] += c

for name, config in configs.items():
    config['vocab_size'] = len(config['chars']) + 3
    with open(f'configs/{name}.json', mode='w') as fp:
        fp.write(json.dumps(config, ensure_ascii=False, indent=2))
