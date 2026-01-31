import argparse
from PIL import Image
from src.scorer import CLIPImageScorer


def parse_args():
    parser = argparse.ArgumentParser(description="Pick preferred image vs a reference image using CLIP")
    parser.add_argument("--ref", required=True, help="Reference image path")
    parser.add_argument("--img_a", required=True, help="Candidate image A")
    parser.add_argument("--img_b", required=True, help="Candidate image B")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def main():
    args = parse_args()
    ref = Image.open(args.ref).convert("RGB")
    img_a = Image.open(args.img_a).convert("RGB")
    img_b = Image.open(args.img_b).convert("RGB")

    scorer = CLIPImageScorer(model_name=args.model, device=args.device)
    score_a = scorer(img_a, ref)
    score_b = scorer(img_b, ref)

    preferred = args.img_a if score_a >= score_b else args.img_b
    print(f"score_a={score_a:.6f} img_a={args.img_a}")
    print(f"score_b={score_b:.6f} img_b={args.img_b}")
    print(f"preferred={preferred}")


if __name__ == "__main__":
    main()
