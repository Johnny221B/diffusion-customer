"""Add role labels above the four unmodified feedback-example images."""
import os
import shutil
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cmts_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'results/pub_fig/discrete_feedback_examples'
BACKUP = ROOT / 'outputs/discrete_feedback_examples_originals'
PANELS = [
    ('a_reference_red.png', '(a) Reference', ''),
    ('b_canvas_seed34_d0.6205.png', '(b) Competitor', ''),
    ('c_below_threshold_red_seed34_d0.4340.png', '(c) Closer to reference', 'than the competitor'),
    ('d_above_threshold_mosaic_seed34_d0.6973.png', '(d) Farther from reference', 'than the competitor'),
]

def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name, title, subtitle in PANELS:
        raw = BACKUP / name
        if not raw.exists():
            shutil.copy2(TARGET / name, raw)
        fig = plt.figure(figsize=(4.96, 5.66), dpi=100, facecolor='white')
        ax = fig.add_axes([0, 0, 1, 496/566])
        with Image.open(raw) as im:
            ax.imshow(im.convert('RGB'))
        ax.axis('off')
        fig.text(.5, .955, title, ha='center', va='center', fontsize=15, fontweight='semibold')
        if subtitle:
            fig.text(.5, .912, subtitle, ha='center', va='center', fontsize=15, fontweight='semibold')
        fig.savefig(TARGET / name, dpi=100, facecolor='white')
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9.92,11.32))
    for ax, (name, _, _) in zip(axes.flat, PANELS):
        with Image.open(TARGET / name) as im:
            assert im.size == (496,566)
            ax.imshow(im)
        ax.axis('off')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=.025,hspace=.025)
    fig.savefig(BACKUP / 'labeled_preview.png',dpi=120)
    for ext in ('png', 'pdf'):
        fig.savefig(TARGET / f'discrete_feedback_examples_combined.{ext}',
                    dpi=300, facecolor='white')
    plt.close(fig)

    # A compact landscape alternative, with vector titles for the PDF.
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, (name, title, subtitle) in zip(axes, PANELS):
        with Image.open(BACKUP / name) as im:
            ax.imshow(im.convert('RGB'))
        ax.axis('off')
        ax.set_title(title + '\n' + (subtitle or ' '),
                     fontsize=10.5, fontweight='semibold', pad=8)
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.81,
                        wspace=.045)
    for ext in ('png', 'pdf'):
        fig.savefig(TARGET / f'discrete_feedback_examples_row.{ext}',
                    dpi=300, bbox_inches='tight', pad_inches=.04,
                    facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    main()
