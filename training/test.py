import json
import sys
import torch
from transformers import BatchEncoding
from model import PartOfSpeechEstimatorPL

model = PartOfSpeechEstimatorPL.load_from_checkpoint(sys.argv[1])
model.model.cpu()

with open('./data/encoded/test.jsonl') as fp:
    token_count = 0
    correctly_estimated_tokens = 0
    correctly_estimated_splits = 0
    appeared_words = 0
    label_words = 0
    estimated_words = 0
    correct_words = 0
    appeared_splitpoints = 0
    label_splitpoints = 0
    estimated_splitpoints = 0
    correct_splitpoints = 0

    for row in fp:
        data = json.loads(row)
        data['attention_mask'] = [1] * len(data['input_ids'])
        data['token_type_ids'] = [0] * len(data['input_ids'])
        bert_input = BatchEncoding({k: [v] for k, v in data.items()}, tensor_type='pt')
        with torch.no_grad():
            predicted_ids = model.forward(**bert_input).tolist()[0]

        token_count += len(data['input_ids'])
        for pos_id, estimated_id  in zip(data['labels'], predicted_ids):
            if pos_id == estimated_id:
                correctly_estimated_tokens += 1
            if (pos_id == 1 and estimated_id == 1) or (pos_id != 1 and estimated_id != 1):
                correctly_estimated_splits += 1

        label_wordset = set()
        label_splitpoint = set()
        prev_i = -1
        prev_pos = -1
        for i, _id in enumerate(data['labels']):
            if _id > 1:
                if prev_i > -1:
                    label_wordset.add((prev_i, i, prev_pos))
                    label_splitpoint.add((prev_i, i))
                prev_i = i
                prev_pos = _id
        label_wordset.add((prev_i, len(data['labels']), prev_pos))
        label_splitpoint.add((prev_i, len(data['labels'])))

        estimated_wordset = set()
        estimated_splitpoint = set()
        prev_i = -1
        prev_pos = -1
        for i, _id in enumerate(predicted_ids):
            if _id > 1:
                if prev_i > -1:
                    estimated_wordset.add((prev_i, i, prev_pos))
                    estimated_splitpoint.add((prev_i, i))
                prev_i = i
                prev_pos = _id
        estimated_wordset.add((prev_i, len(predicted_ids), prev_pos))
        estimated_splitpoint.add((prev_i, len(predicted_ids)))

        appeared = label_wordset | estimated_wordset
        correct = label_wordset & estimated_wordset
        appeared_words += len(appeared)
        correct_words += len(correct)
        label_words += len(label_wordset)
        estimated_words += len(estimated_wordset)
        appeared_splitpoint = label_splitpoint | estimated_splitpoint
        correct_splitpoint = label_splitpoint & estimated_splitpoint
        appeared_splitpoints += len(appeared_splitpoint)
        correct_splitpoints += len(correct_splitpoint)
        label_splitpoints += len(label_splitpoint)
        estimated_splitpoints += len(estimated_splitpoint)

    print('token metrics')
    print(f'POS: {correctly_estimated_tokens}/{token_count} ({correctly_estimated_tokens*100/token_count}%)')
    print(f'TKN: {correctly_estimated_splits}/{token_count} ({correctly_estimated_splits*100/token_count}%)')
    print('pos metrics')
    print(f'IOU: {correct_words}/{appeared_words} ({correct_words*100/appeared_words}%)')
    print(f'PRC: {correct_words}/{estimated_words} ({correct_words*100/estimated_words}%)')
    print(f'RCL: {correct_words}/{label_words} ({correct_words*100/label_words}%)')
    print(f'F1 : { 2 / (estimated_words/correct_words + label_words/correct_words) }')
    print('split metrics')
    print(f'IOU: {correct_splitpoints}/{appeared_splitpoints} ({correct_splitpoints*100/appeared_splitpoints}%)')
    print(f'PRC: {correct_splitpoints}/{estimated_splitpoints} ({correct_splitpoints*100/estimated_splitpoints}%)')
    print(f'RCL: {correct_splitpoints}/{label_splitpoints} ({correct_splitpoints*100/label_splitpoints}%)')
    print(f'F1 : { 2 / (estimated_splitpoints/correct_splitpoints + label_splitpoints/correct_splitpoints) }')
