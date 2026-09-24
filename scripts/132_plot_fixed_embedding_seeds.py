"""Display the first ten pre-existing seeds, without selecting favorable outputs."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cmts_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/pca_reconstruction_review'
fig, axes=plt.subplots(2,5,figsize=(15,6.5))
for seed, ax in enumerate(axes.flat):
    path=ROOT/f'outputs/multiseed_s228_M40_0510_0241/imgs/097_brown_seed{seed:02d}.png'
    ax.imshow(Image.open(path)); ax.set_title(f'Seed {seed}',fontsize=12); ax.axis('off')
fig.suptitle('Fixed embedding, different generation seeds',fontsize=17)
fig.tight_layout(rect=(0,0,1,.95))
for ext in ['png','pdf']:
    fig.savefig(OUT/f'fixed_brown_embedding_10_seeds.{ext}',dpi=200)
plt.close(fig)
