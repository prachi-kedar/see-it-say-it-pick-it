#!/usr/bin/env python3
"""Check step 2 alone: does the scene graph have a pixel on every object in the ground-truth masks,
and does that object have the right noun (gear, cube, mug, ...)?

    python pipeline/check_vision.py --graphs scene_graphs --data dataset

If an object is missed here, no prompt can make Apertus point at it later.
Trays, plates and saucers have no mask (they are never the target), so they are not checked.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
from evaluate import SceneGT  # noqa: E402


def gt_noun(name: str) -> str:
    """red_gear_2 -> gear, small_part -> part"""
    return re.sub(r"_\d+$", "", name).split("_")[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", type=Path, default=Path("outputs/scene_graphs"))
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    args = ap.parse_args()

    gt_dir = args.data / "ground_truth"
    hit = total = named = 0
    for path in sorted(args.graphs.glob("scene_*[0-9].json")):   # not the *.boxes.json caches
        graph = json.loads(path.read_text())
        gt = SceneGT(args.data, gt_dir, graph["scene"])
        on = {}   # ground-truth object -> our objects whose pixel lies on it
        for o in graph["objects"]:
            name = gt.object_at(o["pixel"])
            if name:
                on.setdefault(name, []).append(o)
        print(f"\n{graph['scene']}")
        for name in gt.names.values():
            total += 1
            if name in on:
                hit += 1
                right_name = any(o["noun"] == gt_noun(name) for o in on[name])
                named += right_name
                extra = "" if right_name else f"   WRONG NAME (should be a {gt_noun(name)})"
                extra += "   (more than one detection on it)" if len(on[name]) > 1 else ""
                print(f"  ok      {name:<14} <- {', '.join(o['id'] for o in on[name])}{extra}")
            else:
                print(f"  MISSED  {name}")
        stray = [o["id"] for o in graph["objects"]
                 if o["noun"] not in ("tray", "plate", "saucer") and not gt.object_at(o["pixel"])]
        if stray:
            print(f"  on no object: {', '.join(stray)}")
    print(f"\nobjects found: {hit} of {total}, with the right name: {named} of {total}")


if __name__ == "__main__":
    main()
