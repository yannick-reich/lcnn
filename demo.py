#!/usr/bin/env python3
"""Process an image with the trained neural network
Usage:
    demo.py [options] <yaml-config> <checkpoint> <input-dir> <output-dir>...
    demo.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <input-dir>                   Path to images
   <output-dir>                  Output directory for npz files containing lines and scores

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
"""

import os
import os.path as osp
import pprint
import random
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import torch.mps as mps
import yaml
from docopt import docopt

import lcnn
from lcnn.config import C, M
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.postprocess import postprocess
from lcnn.utils import recursive_to

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def main():
    args = docopt(__doc__)  # parse arguments based on docstring above
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"  # if no yaml config file is specified, use the default
    C.update(C.from_yaml(filename=config_file)) # store config in dict (C is a Box object which is just an extension of dict)
    M.update(C.model)    # get the model part out of config
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # configuring device for torch (cuda if supported (nvidia), mps if apple, cpu otherwise)
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True   # TODO check if I need sth like this in mps
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    elif torch.backends.mps.is_available():
        device_name = "mps"
        print("MPS is enabled")
    else:
        print("using CPU")
    device = torch.device(device_name)
    checkpoint = torch.load(args["<checkpoint>"], map_location=device)

    # Load model
    model = lcnn.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Process images
    img_list = glob.glob(args["<input-dir>"] + "/*", recursive=True)
    for img_path in img_list:
        print(f"Processing {img_path}")
        im = skimage.io.imread(img_path)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        with torch.no_grad():
            input_dict = {
                "image": image.to(device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
                },
                "mode": "testing",
            }
            H = model(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2] # TODO what does this scaling do?
        scores = H["score"][0].cpu().numpy()
        for i in range(1, len(lines)):      # TODO why this cutoff?
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        # TODO save nlines and nscores to npz
        img_name = str(osp.basename(img_path)) # extract image name from path
        np.savez(
            osp.join(args["<output-dir>"][0], img_name.replace(".jpg","")),
            nlines = nlines,
            nscores = nscores
        ) # save lines and scores to npz file


if __name__ == "__main__":
    main()
