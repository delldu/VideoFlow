"""Image/Video Autops Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import torch
import torch.nn.functional as F

from . import network
from . import flow_viz

import todos
import pdb

def get_video_flow_model():
    """Create model."""
    model = network.BOFNet()
    # model = todos.model.ResizePadModel(model)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")
    

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);
    # todos.data.mkdir("output")
    # if not os.path.exists("output/video_flow.torch"):
    #     model.save("output/video_flow.torch")

    return model, device


def flow_predict(input_files, output_dir, horizon=None):
    from tqdm import tqdm

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_video_flow_model()

    # load files
    image_filenames = todos.data.load_files(input_files)
    n_images = total=len(image_filenames)

    # start predict
    progress_bar = tqdm(total=n_images - 2)
    for i in range(1, n_images - 1):
        progress_bar.update(1)

        input_tensor1 = todos.data.load_tensor(image_filenames[i-1])
        input_tensor2 = todos.data.load_tensor(image_filenames[i])
        input_tensor3 = todos.data.load_tensor(image_filenames[i+1])
        B, C, H, W = input_tensor2.size()

        assert input_tensor1.size() == input_tensor2.size() and input_tensor3.size() == input_tensor2.size(), \
            "input tensors must be have same size"

        input_tensors = torch.cat([input_tensor1, input_tensor2, input_tensor3], dim=0)

        with torch.no_grad():
            predict_tensor = model(input_tensors.to(device))

        # todos.debug.output_var("predict_tensor", predict_tensor)
        # for BOF_sintel.pth
        # tensor [flow_pre] size: [2, 2, 436, 1024], min: -34.304611, max: 27.719576, mean: -0.059822
        # for BOF_things.pth
        # tensor [predict_tensor] size: [2, 2, 432, 1024], min: -34.027592, max: 31.108303, mean: -0.055991
        # for BOF_things_288960noise.pth
        # tensor [predict_tensor] size: [2, 2, 432, 1024], min: -34.538429, max: 33.962624, mean: -0.06187

        flow_image1, flow_image2 = flow_viz.vis_pre(predict_tensor.cpu())
        flow_image1 = F.interpolate(flow_image1, (H, W), mode="bilinear", align_corners=False)
        flow_image2 = F.interpolate(flow_image2, (H, W), mode="bilinear", align_corners=False)

        output_filename = f"{output_dir}/{os.path.basename(image_filenames[i])}"
        todos.data.save_tensor([input_tensor1, input_tensor2, input_tensor3, flow_image1, flow_image2], output_filename)
    todos.model.reset_device()

