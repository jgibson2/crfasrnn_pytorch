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
import os
import re
from crfasrnn import util
import numpy as np
from crfasrnn.crfasrnn_model import CrfRnnNet
from crfasrnn.params import DenseCRFParams


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        help="Path to the .pth file (download from https://tinyurl.com/crfasrnn-weights-pth)",
        required=True,
    )
    parser.add_argument("--image", help="Path to the input image", required=False)
    parser.add_argument("--file", help="Path to the input image file", required=False)
    parser.add_argument("--output", help="Path to the output label image OR output filename base", default=None)
    parser.add_argument("--parameters", help="Path to the input parameters")
    args = parser.parse_args()

    if args.parameters:
        params = DenseCRFParams(*np.load(args.parameters).ravel())
    else:
        params = DenseCRFParams()

    model = CrfRnnNet(params=params)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    if args.image:
        img_data, img_h, img_w, size = util.get_preprocessed_image(args.image)
        output_file = args.output or args.image + "_labels.png"

        out = model.forward(torch.from_numpy(img_data))

        probs = out.detach().numpy()[0]
        label_im = util.get_label_image(probs, img_h, img_w, size)
        label_im.save(output_file)

    elif args.file:
        with open(args.file, 'r') as f:
            files = f.readlines()
            for file_pair in files:
                fs = file_pair.split()
                if len(fs) < 1:
                    continue
                outfile = (args.output or '') + re.sub('\.(jpg|png|jpeg|tiff|gif)', '',
                                                   fs[0].split(os.path.sep)[-1]) + '_labels.png'
                print(f'Predicting for {outfile}...', flush=True)
                img_data, img_h, img_w, size = util.get_preprocessed_image(fs[0])
                out = model.forward(torch.from_numpy(img_data))
                probs = out.detach().numpy()[0]
                label_im = util.get_label_image(probs, img_h, img_w, size)
                label_im.save(outfile)
                del out
                del probs
                del label_im



if __name__ == "__main__":
    main()
