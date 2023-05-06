import os
import torch
import argparse
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from transformers import pipeline, AutoTokenizer, VisualBertModel

def diffuse(model, prompt: str):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='GenerateImages',
                description='Generates images from a given dataset in to a given folder.')
    #
    parser.add_argument('-d', '--dataset',
               type=str,
               required=True,
               choices=['sst2', 'mrpc', 'cola', 'wnli', 'rte'],
               help='The dataset to use.')
    parser.add_argument('-s', '--split',
               type=str,
               choices=['train', 'test', 'validation'],
               default='train',
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
    parser.add_argument('-p', '--prompt',
               type=str,
               default='',
               help='Prompt to append to each text sentence. Ex: realistic, mood, cartoon, etc.')
    parser.add_argument('-w', '--weights',
               type=str,
               default='stabilityai/stable-diffusion-2-1-base',
               help='Huggingface diffusion weights.')

    args = parser.parse_args()
    print(args)

    #loading the model.
    model = StableDiffusionPipeline.from_pretrained(args.weights, torch_dtype=torch.float16)
    model.to("cuda")

    #loading the dataset
    dataset = load_dataset('glue', args.dataset, split=args.split)

    start = 0
    end = len(dataset) 
    if args.start is None and args.start is None:
        start = 0
        end = len(dataset) 
    elif args.start >= 0 and args.end > args.start and args.end <= end:
        start = args.start
        end = args.end

    for sample in range(start,end):
        if args.dataset == 'wnli':
            prompt = dataset[sample]['sentence1']
            prompt2 = dataset[sample]['sentence2']
            label = dataset[sample]['label']
            idx = dataset[sample]['idx']
            output_file = 'wnli_' + str(idx) + '.jpg'
            output_path = os.path.join(args.output, output_file)
            if os.path.isfile(output_path):
                continue
            image = model(prompt + ' ' + prompt2).images[0]
            print(output_path)
            image.save(output_path)
        elif args.dataset == 'rte':
            prompt = dataset[sample]['sentence1']
            prompt2 = dataset[sample]['sentence2']
            label = dataset[sample]['label']
            idx = dataset[sample]['idx']
            output_file = 'rte_' + str(idx) + '.jpg'
            output_path = os.path.join(args.output, output_file)
            if os.path.isfile(output_path):
                continue
            image = model(prompt + ' ' + prompt2).images[0]
            print(output_path)
            image.save(output_path)
        elif args.dataset == 'sst2':
            prompt = dataset[sample]['sentence']
            label = dataset[sample]['label']
            idx = dataset[sample]['idx']
            output_file = 'sst_' + str(idx) + '.jpg'
            output_path = os.path.join(args.output, output_file)
            if os.path.isfile(output_path):
                continue
            image = model(prompt).images[0]
            print(output_path)
            image.save(output_path)
        elif args.dataset == 'mrpc':
            prompt = dataset[sample]['sentence1']
            prompt2 = dataset[sample]['sentence2']
            label = dataset[sample]['label']
            idx = dataset[sample]['idx']
            output_file = 'mrpc_' + str(idx) + '.jpg'
            output_path = os.path.join(args.output, output_file)
            if os.path.isfile(output_path):
                continue
            image = model(prompt + ' ' + prompt2).images[0]
            print(output_path)
            image.save(output_path)
        elif args.dataset == 'cola':
            prompt = dataset[sample]['sentence']
            label = dataset[sample]['label']
            idx = dataset[sample]['idx']
            output_file = 'cola_' + str(idx) + '.jpg'
            output_path = os.path.join(args.output, output_file)
            if os.path.isfile(output_path):
                continue
            image = model(prompt).images[0]
            print(output_path)
            image.save(output_path)
