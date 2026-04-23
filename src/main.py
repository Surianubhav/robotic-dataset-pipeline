from pipeline.processor import FrameProcessor
from pipeline.processor import run_video, run_live

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["video", "live"], default="live")
    parser.add_argument("--input", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "video":
        run_video(args.input)
    else:
        run_live(0)
