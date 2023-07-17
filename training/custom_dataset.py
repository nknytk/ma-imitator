import json
from torch.utils.data import Dataset
from transformers import BatchEncoding


class CustomDataset(Dataset):
    def __init__(self, data_file_path: str, max_length: int):
        self.data = []
        with open(data_file_path) as fp:
            for row in fp:
                self.data.append(json.loads(row))
        self.max_length = max_length
        self.token_type_ids = [0] * max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.data[index]['input_ids']
        labels = self.data[index]['labels']
        attention_mask = [1] * len(input_ids)
        zero_padding = [0] * (self.max_length - len(input_ids))
        item = {
            'input_ids': input_ids + zero_padding,
            'labels': labels + zero_padding,
            'attention_mask': [1] * len(input_ids) + zero_padding,
            'token_type_ids': self.token_type_ids
        }

        return BatchEncoding(data=item, tensor_type='pt')
        
