# import os
# from huggingface_hub import snapshot_download

# # 常见写法：SD 3.5 Large 的官方仓库通常在 stabilityai 名下
# repo_id = "stabilityai/stable-diffusion-3.5-large"   # 如果你看到的repo名不一样，就用你实际的那个

# local_dir = "/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large"

# snapshot_download(
#     repo_id=repo_id,
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,
#     token=os.environ.get("HF_TOKEN"),
#     # 建议加上：避免下载无关文件（如果repo里有大杂烩）
#     # allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md", "tokenizer/*", "text_encoder/*", "vae/*", "transformer/*"],
# )
# print("Downloaded to:", local_dir)

import os
from huggingface_hub import snapshot_download

repo_id = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
local_dir = "/home/linyuliu/jxmount/diffusion_custom/models/controlnets/sd35_large_controlnet_canny"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print("Downloaded to:", local_dir)
