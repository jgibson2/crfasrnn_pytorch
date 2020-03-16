"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse

import torch

from crfasrnn import util
from PIL import Image
import numpy as np
from crfasrnn.crfasrnn_model import CrfRnnNet
from crfasrnn.params import DenseCRFParams
from optimize_parameters import *

def compute_accuracy(params, model, img_data, img_h, img_w, size, ground_truth):
    model.set_params(params)

    out = model.forward(torch.from_numpy(img_data))

    probs = out.detach().numpy()[0]
    label_im = util.get_label_image(probs, img_h, img_w, size)

    predicted = np.asarray(label_im.convert('RGB'))

    acc = util.compute_jaccard_index(predicted, ground_truth)

    del predicted
    del label_im
    del out
    del probs

    return acc

def compute_average_accuracy(params, model, imgs, img_hs, img_ws, sizes, ground_truths):
    model.set_params(params)
    acc = 0.0
    for img_data, img_h, img_w, size, ground_truth in zip(imgs, img_hs, img_ws, sizes, ground_truths):
        out = model.forward(torch.from_numpy(img_data))

        probs = out.detach().numpy()[0]
        label_im = util.get_label_image(probs, img_h, img_w, size)

        predicted = np.asarray(label_im.convert('RGB'))

        acc += util.compute_jaccard_index(predicted, ground_truth)

        del predicted
        del label_im
        del out
        del probs

    return acc / len(imgs)

def run_single_image(args):
    img_data, img_h, img_w, size = util.get_preprocessed_image(args.image)

    output_file = args.output + "_labels.png"

    ground_truth_image = Image.open(args.gt).convert('RGB')
    ground_truth = np.asarray(ground_truth_image)

    model = CrfRnnNet()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    if args.parameters:
        best_params = np.load(args.parameters)
    else:
        scales = np.array([50.0, 1.0, 1.0, 1.0, 1.0])

        acc = lambda x: compute_accuracy(DenseCRFParams(*(scales.ravel() * x)), model, img_data, img_h, img_w, size,
                                         ground_truth)

        bounds = np.array([
            [0.01, 10.0],
            [0.01, 10.0],
            [0.01, 10.0],
            [0.01, 10.0],
            [0.01, 10.0]
        ])

        params, losses = bayesian_optimisation(args.iterations, acc, bounds=bounds, n_pre_samples=5)
        params *= scales.reshape(1, -1)

        best_params_idx = np.argmax(losses)
        best_params = params[best_params_idx]

        print(f'Got best params {params[best_params_idx]} with loss {losses[best_params_idx]}')

        np.save(args.output, params[best_params_idx])

    model.set_params(DenseCRFParams(*best_params))

    out = model.forward(torch.from_numpy(img_data))

    probs = out.detach().numpy()[0]
    label_im = util.get_label_image(probs, img_h, img_w, size)
    label_im.save(output_file)


def run_multiple_images(args):
    imgs = []
    img_hs = []
    img_ws = []
    sizes = []
    gts = []
    with open(args.file, 'r') as f:
        files = f.readlines()
        for file_pair in files:
            fs = file_pair.split()
            if len(fs) != 2:
                continue
            img_data, img_h, img_w, size = util.get_preprocessed_image(fs[0])
            imgs.append(img_data)
            img_hs.append(img_h)
            img_ws.append(img_w)
            sizes.append(size)
            ground_truth_image = Image.open(fs[1]).convert('RGB')
            ground_truth = np.asarray(ground_truth_image)
            gts.append(ground_truth)

    output_file = args.output + "_labels.png"

    model = CrfRnnNet()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    scales = np.array([50.0, 1.0, 1.0, 1.0, 1.0])

    acc = lambda x: compute_average_accuracy(DenseCRFParams(*(scales.ravel() * x)), model, imgs, img_hs, img_ws, sizes,
                                     gts)

    bounds = np.array([
        [0.01, 10.0],
        [0.01, 10.0],
        [0.01, 10.0],
        [0.01, 10.0],
        [0.01, 10.0]
    ])

    params, losses = bayesian_optimisation(args.iterations, acc, bounds=bounds, n_pre_samples=5)
    params *= scales.reshape(1, -1)

    best_params_idx = np.argmax(losses)

    print(f'Got best params {params[best_params_idx]} with loss {losses[best_params_idx]}')

    np.save(args.output, params[best_params_idx])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        help="Path to the .pth file (download from https://tinyurl.com/crfasrnn-weights-pth)",
        required=True,
    )
    parser.add_argument("--image", help="Path to the input image")
    parser.add_argument("--gt", help="Path to the ground truth image")
    parser.add_argument("--output", help="Path to the output parameters", default='out')
    parser.add_argument("--parameters", help="Path to the input parameters")
    parser.add_argument("--iterations", help="Number of iterations for Bayesian opt", type=int, default=25)
    parser.add_argument('--file', help='File of images and ground truths', type=str)


    args = parser.parse_args()

    if(args.file):
        run_multiple_images(args)
    else:
        run_single_image(args)

if __name__ == "__main__":
    main()
