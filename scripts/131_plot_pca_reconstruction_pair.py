"""Arrange unchanged model outputs into a two-panel reconstruction figure."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cmts_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/pca_reconstruction_review'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=18)
    parser.add_argument('--contact', action='store_true')
    args = parser.parse_args()
    if args.contact:
        fig, axes = plt.subplots(4,2,figsize=(8,16))
        for seed, row in zip([1,13,17,18], axes):
            for kind, ax in zip(['normal','pca16'], row):
                ax.imshow(Image.open(OUT / f'brown_seed{seed:02d}_{kind}.png'))
                ax.set_title(f'{seed}: {kind}'); ax.axis('off')
        fig.tight_layout(); fig.savefig(OUT / 'paired_contact.png',dpi=100)
    else:
        plt.rcParams['pdf.fonttype'] = 42
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        for kind, title, ax in zip(['normal','pca16'], ['Normal reconstruction','PCA-compressed reconstruction'], axes):
            ax.imshow(Image.open(OUT / f'brown_seed{args.seed:02d}_{kind}.png'))
            ax.set_title(title, fontsize=17, pad=15); ax.axis('off')
        fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.88,wspace=.035)
        for ext in ['png','pdf']:
            fig.savefig(OUT / f'normal_vs_pca_reconstruction.{ext}',dpi=300,facecolor='white')
        (OUT / 'selected_pair.json').write_text(json.dumps(dict(word='brown',seed=args.seed,pca_dim=16,selection='Illustrative pair selected for full shoe visibility and visual similarity; original render pixels retained.'),indent=2))
    plt.close(fig)

if __name__ == '__main__':
    main()
