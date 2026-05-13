import os
import re
import math
from PIL import Image, ImageDraw, ImageFont


def parse_epoch(folder_name):
    m = re.match(r"epoch_(\d+)$", folder_name)
    if m:
        return int(m.group(1))
    return None


def parse_idx(filename):
    m = re.search(r"_idx(\d+)_", filename)
    if m:
        return int(m.group(1))
    return None


def add_label(img, text, label_height=30):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + label_height), "white")
    canvas.paste(img, (0, label_height))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (w - text_w) // 2
    y = (label_height - text_h) // 2
    draw.text((x, y), text, fill="black", font=font)

    return canvas


def merge_images_grid(
    image_infos,
    save_path,
    images_per_row=4,
    bg_color="white",
    spacing=10,
    add_epoch_label=True
):
    """
    image_infos: [(epoch, image_path), ...]
    按网格拼图，每行 images_per_row 张，最后一行不足也支持
    """
    opened = []
    for epoch, path in image_infos:
        img = Image.open(path).convert("RGB")
        if add_epoch_label:
            img = add_label(img, f"epoch_{epoch:05d}")
        opened.append(img)

    if not opened:
        return

    # 假设所有图片尺寸一致；如果不一致，也按最大宽高来放
    cell_w = max(img.width for img in opened)
    cell_h = max(img.height for img in opened)

    n = len(opened)
    n_rows = math.ceil(n / images_per_row)
    n_cols = min(images_per_row, n)

    canvas_w = n_cols * cell_w + (n_cols - 1) * spacing
    canvas_h = n_rows * cell_h + (n_rows - 1) * spacing

    merged = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    for i, img in enumerate(opened):
        row = i // images_per_row
        col = i % images_per_row

        x = col * (cell_w + spacing)
        y = row * (cell_h + spacing)

        # 居中贴到单元格里
        paste_x = x + (cell_w - img.width) // 2
        paste_y = y + (cell_h - img.height) // 2
        merged.paste(img, (paste_x, paste_y))

    merged.save(save_path)
    print(f"Saved: {save_path}")


def main(root_dir, output_dir, step=30, include_epoch1=False, images_per_row=4):
    merge_dir = os.path.join(output_dir, "merge")
    os.makedirs(merge_dir, exist_ok=True)

    # 找所有 epoch 文件夹
    epoch_folders = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path):
            epoch_num = parse_epoch(name)
            if epoch_num is not None:
                epoch_folders.append((epoch_num, full_path))

    epoch_folders.sort(key=lambda x: x[0])

    if not epoch_folders:
        print("没有找到任何 epoch_xxxxx 文件夹")
        return

    # 选每 step 个 epoch
    selected_epochs = []
    for epoch_num, folder_path in epoch_folders:
        if epoch_num % step == 0:
            selected_epochs.append((epoch_num, folder_path))

    # 可选：把 epoch_00001 也加进去
    if include_epoch1:
        for epoch_num, folder_path in epoch_folders:
            if epoch_num == 1:
                if (epoch_num, folder_path) not in selected_epochs:
                    selected_epochs.insert(0, (epoch_num, folder_path))
                break

    if not selected_epochs:
        print(f"没有找到满足每 {step} epoch 的文件夹")
        return

    print("选中的 epoch：", [e for e, _ in selected_epochs])

    # 收集每个 idx 的图片
    idx_to_images = {}
    for epoch_num, folder_path in selected_epochs:
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue

            idx = parse_idx(fname)
            if idx is None:
                continue

            img_path = os.path.join(folder_path, fname)
            idx_to_images.setdefault(idx, []).append((epoch_num, img_path))

    if not idx_to_images:
        print("没有找到任何带 idx 的图片")
        return

    # 每个 idx 分别拼图
    for idx, image_infos in idx_to_images.items():
        image_infos.sort(key=lambda x: x[0])

        save_name = f"idx_{idx}_merged.png"
        save_path = os.path.join(merge_dir, save_name)

        merge_images_grid(
            image_infos=image_infos,
            save_path=save_path,
            images_per_row=images_per_row,
            bg_color="white",
            spacing=10,
            add_epoch_label=True
        )

    print(f"\n全部完成，结果保存在: {merge_dir}")


if __name__ == "__main__":
    # ===== 改成你的路径 =====
    root_dir = "/home/linyuliu/jxmount/diffusion_custom/outputs/R100_v24_0317_0822"
    # output_dir = "/home/linyuliu/jxmount/diffusion_custom/outputs/R10_v24_0317_1306"

    main(
        root_dir=root_dir,
        output_dir=root_dir,
        step=10,
        include_epoch1=True,
        images_per_row=4
    )
