import json
import random
import sys
import fugashi
import preprocess


def create(config_file_path: str, data_file_paths: str, output_file_path: str):
    tagger = fugashi.Tagger()
    preprocessor = preprocess.Preprocessor(config_file_path)

    target_length = random.randint(1, preprocessor.max_length)
    target_text = ''

    wp = open(output_file_path, mode='w')
    for data_file_path in data_file_paths.split(','):
        with open(data_file_path) as fp:
            for row in fp:
                row = preprocess.normalize(row)
                current_length = len(target_text)
                rest_length = target_length - current_length
                if len(row) < rest_length:
                    target_text += row
                elif rest_length < len(row):
                    target_text += row[:rest_length]
                    target_text = target_text.strip()
                    encoded_input = preprocessor.encode_training_input(tagger, target_text, do_padding=False)
                    # 元に戻せることを確認する
                    decoded_tokens = preprocessor.decode(target_text, encoded_input['labels'])
                    _txt = ''.join(t[0] for t in decoded_tokens).strip()
                    if _txt == target_text:
                        wp.write(json.dumps(encoded_input))
                        wp.write('\n')
                    else:
                        print(target_text)
                        print(_txt)
                        print('-----')

                    row = row[rest_length:]
                    target_length = random.randint(1, preprocessor.max_length)
                    target_text = row if len(row) < target_length else ''

    wp.close()


if __name__ == '__main__':
    create(sys.argv[1], sys.argv[2], sys.argv[3])
