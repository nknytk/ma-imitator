import sys
import torch
import onnx
from model import PartOfSpeechEstimatorPL


max_length, input_model_path, output_onnx_path = sys.argv[1:4]
max_length = int(max_length)
model = PartOfSpeechEstimatorPL.load_from_checkpoint(input_model_path)
sample_data = {
    'input_ids': [0] * max_length,
    'attention_mask': [1] * max_length,
    'token_type_ids': [0] * max_length
}
sample_input = (
    torch.tensor([sample_data['input_ids']]),
    torch.tensor([sample_data['attention_mask']]),
    torch.tensor([sample_data['token_type_ids']])
)
model.to_onnx(output_onnx_path, sample_input, export_params=True, opset_version=18)
model = onnx.load(output_onnx_path)
print('Model Graph:\n\n{}'.format(onnx.helper.printable_graph(model.graph)))
