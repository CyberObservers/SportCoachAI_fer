import torch
from models import VGG, FaceCNN, DenseNet121

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
device = 'cpu'
model = torch.load('./weights/vgg_it100.pkl', map_location=device)

batch_size = 1
input_shape = (1, 48, 48)
x = torch.randn(batch_size, *input_shape).to(device)
print(x.shape)
export_onnx_file = './weights/vgg_it100.onnx'
torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=9,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  export_params=True,
                  dynamic_axes={"input": {0: "batch_size"},
                                "output": {0: "batch_size"}})
