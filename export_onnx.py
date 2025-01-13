import time
import torch
from pathlib import Path
import onnxruntime as ort

if __name__ == '__main__':
    fp = Path('run_sp2.0/cifar10/prune/')
    fns = ['cifar10_random_resnet56.pth', 'cifar10_l1_resnet56.pth',
           'cifar10_slim_resnet56.pth', 'cifar10_growing_reg_resnet56.pth', 'cifar10_group_sl_resnet56.pth'
           ]
    # fp = Path('./')
    # fns = ['cifar10_pretrained_resnet56.pth']
    for fn in fns:
        name = fn.split('.')[0]
        model = torch.load(str(fp / fn))
        input_tensor = torch.randn(1, 3, 32, 32).to('cuda')
        torch.onnx.export(model, input_tensor, f"{name}.onnx", export_params=True)
        print()

        repeats = 1000
        # warm up
        for _ in range(repeats):
            output = model(input_tensor)
        start = time.time()
        for _ in range(repeats):
            output = model(input_tensor)
        print("PyTorch Inference Time:", time.time() - start)

        session = ort.InferenceSession(f"{name}.onnx")
        input_numpy = input_tensor.cpu().numpy()
        onnx_input = {session.get_inputs()[0].name: input_numpy}
        start = time.time()
        for _ in range(repeats):
            output = session.run(None, onnx_input)
        print("ONNX Inference Time:", time.time() - start)
