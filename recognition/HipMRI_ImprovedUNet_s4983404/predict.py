"""Load actual saved weights; save native-grid predictions and FP/FN panels."""
from pathlib import Path
import argparse
import json
import logging
import time
import nibabel as nib
import numpy as np
import torch
from dataset import ROOT, HipMRISlices
from modules import build_model


def panel(image: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
          destination: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(12, 4))
    for axis in axes:
        axis.imshow(image.T, cmap='gray', origin='lower')
        axis.axis('off')
    axes[1].imshow(np.ma.masked_where(truth.T == 0, truth.T), cmap='tab10',
                   vmin=0, vmax=5, alpha=0.6, origin='lower')
    axes[2].imshow(np.ma.masked_where(prediction.T == 0, prediction.T), cmap='tab10',
                   vmin=0, vmax=5, alpha=0.6, origin='lower')
    overlay = np.zeros((*truth.T.shape, 4))
    overlay[(prediction.T == 5) & (truth.T != 5)] = [1, 0, 0, 0.8]
    overlay[(prediction.T != 5) & (truth.T == 5)] = [0, 0.6, 1, 0.8]
    axes[3].imshow(overlay, origin='lower')
    for axis, title in zip(axes, ['Input', 'Ground truth (IDs 0-5)', 'Prediction',
                                 'Label 5: red FP / blue FN']):
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validate', 'test'], default='validate')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    config = checkpoint['config']
    model = build_model(config['model'], config['width']).to(args.device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    dataset = HipMRISlices(args.root, args.split, args.count)
    for index, (image, target, name) in enumerate(dataset):
        with torch.inference_mode():
            predicted = model(image.unsqueeze(0).to(args.device)).argmax(1)[0].cpu().numpy()
        source = nib.load(dataset.pairs[index][0])
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(predicted.astype(np.uint8), source.affine, header),
                 args.output/name.replace('case_', 'pred_', 1))
        panel(image[0].numpy(), target.argmax(0).numpy(), predicted,
              args.output/f'{name.removesuffix(".nii.gz")}.png')
        logging.info('Predicted %s', name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
