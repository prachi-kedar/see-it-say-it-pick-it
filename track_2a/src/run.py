#!/usr/bin/env python3
"""One command: a folder of scenes -> a results file. Everything runs locally.

    python run.py --data dataset --out results.json
    python run.py --data heldout --out results_heldout.json

Step 2 (vision):  Grounding DINO + CLIP + OpenCV turn every rgb.png into a scene graph
                  (scene_graphs/<folder name>/<scene>.json, with a picture to check by eye).
Step 3 (Apertus): Apertus 1.5 8B, served by Ollama, reads each scene graph and instruction and
                  answers with a plan, a question or a refusal.

The vision models and Apertus do not fit on a 6 GB GPU together, so Apertus is unloaded from
Ollama before the vision step and loaded again (automatically) for step 3.
"""
from __future__ import annotations

import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # src/
sys.path.insert(0, str(ROOT / "pipeline"))
from plan import BASE_URL, MODEL, unload  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder of scene folders, each with rgb.png and instructions.txt")
    ap.add_argument("--out", type=Path, required=True, help="results file to write")
    ap.add_argument("--graphs", type=Path, help="where to keep the scene graphs (default scene_graphs/<data folder name>)")
    ap.add_argument("--base-url", default=BASE_URL, help="Ollama server")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--team", default="see-it-say-it-pick-it")
    args = ap.parse_args()
    graphs = args.graphs or ROOT.parent / "outputs" / "scene_graphs" / args.data.resolve().name

    unload(args.base_url, args.model)
    py = sys.executable
    subprocess.run([py, str(ROOT / "pipeline" / "vision.py"), "--data", str(args.data), "--out", str(graphs)], check=True)
    subprocess.run([py, str(ROOT / "pipeline" / "plan.py"), "--data", str(args.data), "--graphs", str(graphs),
                    "--out", str(args.out), "--base-url", args.base_url, "--model", args.model, "--team", args.team], check=True)


if __name__ == "__main__":
    main()
