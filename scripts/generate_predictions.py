import os
import torch
import ipyplot
import argparse
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='GenerateImages',
                description='Generates predictions from images in a directory into a given folder.')
    #
    parser.add_argument('-i', '--input',
               type=str,
               required=True,
               help='The dataset to use.')
    parser.add_argument('-o', '--output',
               type=str,
               required=True,
               help='Output directory.')
    parser.add_argument('-b', '--start',
               type=int,
               required=False,
               help='Number of starting index in dataset.')
    parser.add_argument('-e', '--end',
               type=int,
               required=False,
               help='Number of ending index in dataset.')
    
    args = parser.parse_args()
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                             rpn_post_nms_top_n_test=1000,
                                                             rpn_pre_nms_top_n_test=1000,
                                                             rpn_score_thresh=.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    start = 0
    end = len(os.listdir(args.input)) 
    if args.start is None and args.start is None:
        start = 0
        end = end
    elif args.start >= 0 and args.end > args.start and args.end <= end:
        start = args.start
        end = args.end
    
    with torch.no_grad():
        for file in sorted(os.listdir(args.input))[start:end]:
            path = os.path.join(args.input, file)
            name, ext = os.path.splitext(file)
            final_path = os.path.join(args.output, name + '.pt')
            if os.path.isfile(final_path):
                continue

            img = Image.open(path)
            to_tensor = torchvision.transforms.ToTensor()
            tensor = to_tensor(img)
            tensor = tensor.unsqueeze(0).to(device)

            predictions = model(tensor)

            print('Writing to file: ', final_path)
            torch.save(predictions, final_path)
        