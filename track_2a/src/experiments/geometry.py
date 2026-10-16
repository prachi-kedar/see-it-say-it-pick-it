#!/usr/bin/env python3
"""Can Apertus do the geometry itself? A side experiment for the report, not part of the pipeline.

    python experiments/geometry.py                    # all variants, the 24 relative + multistep instructions
    python experiments/geometry.py --variants plain examples --limit 3

Apertus gets the scene graph of step 2 and one instruction and names the first object the robot must
pick. That object's pixel is checked against the ground-truth masks, as starter/evaluate.py does. The
pipeline's code gets all 24 right from the same scene graphs, so every miss here is Apertus's.

Variants (what changes against "plain", the v1 way: the object list with pixels and boxes):
  examples   + four worked examples on an invented scene (no dev-set instruction in them)
  vote       five samples at temperature 0.7, the most common answer wins
  facts      + the geometry written out as text: order from left to right, distances, what lies on what
  and the combinations listed in VARIANTS.

Every reply is cached in experiments/geometry_cache.json, so a stopped run carries on where it stopped.
Results: experiments/geometry_results.json (every answer) and a table on stdout.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, sys, urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # track_2a/
SRC = ROOT / "src"
sys.path.insert(0, str(SRC / "pipeline"))
sys.path.insert(0, str(SRC / "starter"))
from plan import MODEL, Scene  # noqa: E402
from evaluate import SceneGT, load_ground_truth  # noqa: E402

VARIANTS = {   # name -> (worked examples, geometry as text, samples)
    "plain": (False, False, 1),
    "examples": (True, False, 1),
    "vote": (False, False, 5),
    "examples+vote": (True, False, 5),
    "facts": (False, True, 1),
    "facts+examples+vote": (True, True, 5),
}
VOTE_TEMPERATURE = 0.7
MAX_TOKENS = 800   # a reply cut off here does not parse and counts as a miss (one ran on for 15 minutes)

TASK = """You control a robot arm above a table, seen from above by a camera. You get a list of the
objects in the photo and one instruction. Answer with the id of the FIRST object the robot must pick
to carry out the instruction.

- x grows to the right and y grows downward, as seen in the photo. "left" means smaller x.
- "box" is [x_left, y_top, x_right, y_bottom]. Two objects are close when their boxes are close.
- The gripper holds one object at a time. If another object lies on the object to move, that object
  must be picked first, so it is the answer.
- Trays, the plate and the saucer are places to put things. They are never the answer.
- The object an instruction measures from ("closest to the mug") is never the answer itself.

Write "analysis" first: the reasoning, step by step, with the numbers you compare. Then "target"."""

EXAMPLE_SCENE = {"objects": [
    {"id": "grey_tray", "noun": "tray", "colour": "grey", "pixel": [700, 1000], "box": [100, 300, 1300, 1700], "inside": None, "on_top_of": []},
    {"id": "blue_cube", "noun": "cube", "colour": "blue", "pixel": [400, 900], "box": [320, 820, 480, 980], "inside": "grey_tray", "on_top_of": []},
    {"id": "green_gear_1", "noun": "gear", "colour": "green", "pixel": [800, 1000], "box": [680, 880, 920, 1120], "inside": "grey_tray", "on_top_of": []},
    {"id": "green_mug", "noun": "mug", "colour": "green", "pixel": [1100, 1000], "box": [1000, 900, 1200, 1100], "inside": "grey_tray", "on_top_of": []},
    {"id": "green_gear_2", "noun": "gear", "colour": "green", "pixel": [1600, 950], "box": [1480, 830, 1720, 1070], "inside": None, "on_top_of": []},
    {"id": "blue_part", "noun": "part", "colour": "blue", "pixel": [1600, 940], "box": [1550, 890, 1650, 990], "inside": None, "on_top_of": ["green_gear_2"]},
    {"id": "banana", "noun": "banana", "colour": "yellow", "pixel": [2000, 1300], "box": [1850, 1200, 2150, 1400], "inside": None, "on_top_of": []},
]}
EXAMPLES = [
    ("pick up the object closest to the mug",
     "The mug (green_mug) is the reference, so it is not the answer. Its box is x 1000-1200. green_gear_1 "
     "ends at x 920: 80 px to the left. blue_cube ends at x 480: 520 px away. green_gear_2 starts at x 1480: "
     "280 px away. The smallest gap is green_gear_1, and nothing lies on it.", "green_gear_1"),
    ("pick up the green gear on the right",
     "Green gears: green_gear_1 at x 800, green_gear_2 at x 1600. The right one has the larger x: green_gear_2. "
     "But blue_part lies on green_gear_2, so blue_part must be picked first.", "blue_part"),
    ("pick up the second object from the left",
     "Pickable objects by x: blue_cube 400, green_gear_1 800, green_mug 1100, green_gear_2 1600, blue_part "
     "1600, banana 2000. The second from the left is green_gear_1, and nothing lies on it.", "green_gear_1"),
    ("move everything from the grey tray to the table except the cube",
     "Inside grey_tray: blue_cube, green_gear_1, green_mug. The cube stays, so green_gear_1 and green_mug are "
     "moved. Nothing lies on either; the first one, from the left, is green_gear_1.", "green_gear_1"),
]


def objects_text(graph: dict) -> str:
    lines = []
    for o in graph["objects"]:
        extra = (f", inside {o['inside']}" if o.get("inside") else "") + \
                (f", lies on {', '.join(o['on_top_of'])}" if o.get("on_top_of") else "")
        lines.append(f"- {o['id']}: {o['colour']} {o['noun']}, pixel x {o['pixel'][0]} y {o['pixel'][1]}, "
                     f"box {o['box']}{extra}")
    return "Objects:\n" + "\n".join(lines)


def facts_text(graph: dict) -> str:
    """The geometry worked out in code and written as sentences, so Apertus only has to read it."""
    scene = Scene(json.loads(json.dumps(graph)))
    pick = sorted(scene.pickable, key=lambda o: o["pixel"][0])
    lines = ["Pickable objects from left to right: " + ", ".join(o["id"] for o in pick) + "."]
    for noun in sorted({o["noun"] for o in pick}):
        same = [o["id"] for o in pick if o["noun"] == noun]
        if len(same) > 1:
            lines.append(f"{noun}s from left to right: {', '.join(same)}.")
    for o in graph["objects"]:
        if o["noun"] == "tray":
            inside = [p["id"] for p in pick if p.get("inside") == o["id"]]
            lines.append(f"In {o['id']}: {', '.join(inside) or 'nothing'}.")
            continue
        others = sorted((p for p in scene.pickable if p is not o), key=lambda p: scene.gap(o, p))
        tops = [p["id"] for p in scene.tops(o)]
        lines.append(f"{o['id']}: " + (f"{', '.join(tops)} lies on it; " if tops else "nothing lies on it; ") +
                     "nearest first: " + ", ".join(f"{p['id']} ({max(0, round(scene.gap(o, p)))} px)" for p in others) + ".")
    return objects_text(graph) + "\n\nFacts (measured from the photo):\n" + "\n".join(f"- {l}" for l in lines)


def system_prompt(examples: bool, facts: bool) -> str:
    if not examples:
        return TASK
    scene = facts_text(EXAMPLE_SCENE) if facts else objects_text(EXAMPLE_SCENE)
    worked = "\n\n".join(f'Instruction: "{i}"\n{{"analysis": "{a}", "target": "{t}"}}' for i, a, t in EXAMPLES)
    return f"{TASK}\n\nWorked examples, all on this invented scene:\n{scene}\n\n{worked}"


def ask(base_url: str, model: str, system: str, user: str, ids: list[str], temperature: float, seed: int) -> str:
    schema = {"type": "object", "required": ["analysis", "target"],
              "properties": {"analysis": {"type": "string"}, "target": {"type": "string", "enum": ids}}}
    body = {"model": model, "stream": False, "format": schema,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192, "num_predict": MAX_TOKENS},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "dataset")
    ap.add_argument("--graphs", type=Path, default=ROOT / "docs" / "dev_runs" / "scene_graphs_v3")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, help="only the first N instructions")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cache", type=Path, default=ROOT / "src" / "experiments" / "geometry_cache.json")
    ap.add_argument("--out", type=Path, default=ROOT / "src" / "experiments" / "geometry_results.json")
    args = ap.parse_args()

    gt = load_ground_truth(args.data / "ground_truth")
    with (args.data / "instructions.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["category"] in ("relative", "multistep")][:args.limit]
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    masks, results = {}, {}

    for name in args.variants:
        examples, facts, samples = VARIANTS[name]
        system = system_prompt(examples, facts)
        results[name] = []
        for r in rows:
            iid, graph = r["instruction_id"], json.loads((args.graphs / f"{r['scene']}.json").read_text())
            scene = Scene(graph)
            ids = [o["id"] for o in scene.pickable]
            user = (facts_text(graph) if facts else objects_text(graph)) + f'\n\nInstruction: "{r["instruction"]}"'
            votes, prompt = [], hashlib.sha1((args.model + system + user).encode()).hexdigest()[:10]
            for s in range(samples):
                key = f"{name}|{iid}|{s}|{prompt}"   # a changed prompt is asked again
                if key not in cache:
                    temperature = 0 if samples == 1 else VOTE_TEMPERATURE
                    cache[key] = ask(args.base_url, args.model, system, user, ids, temperature, seed=s + 1)
                    args.cache.parent.mkdir(parents=True, exist_ok=True)
                    args.cache.write_text(json.dumps(cache, indent=1))
                try:
                    votes.append(json.loads(cache[key])["target"])
                except (json.JSONDecodeError, KeyError):
                    votes.append(None)
            choice = Counter(v for v in votes if v).most_common(1)[0][0] if any(votes) else None
            if r["scene"] not in masks:
                masks[r["scene"]] = SceneGT(args.data, args.data / "ground_truth", r["scene"])
            on = masks[r["scene"]].object_at(scene.by_id[choice]["pixel"]) if choice else None
            hit = on in gt[iid]["targets"]
            results[name].append({"instruction_id": iid, "category": r["category"], "instruction": r["instruction"],
                                  "votes": votes, "answer": choice, "pixel_on": on, "correct": gt[iid]["targets"],
                                  "hit": hit, "analysis": cache[f"{name}|{iid}|0|{prompt}"][:1500]})
            print(f"{name:20s} {iid:12s} {'ok  ' if hit else 'MISS'} {choice} {votes if samples > 1 else ''}", flush=True)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n{'variant':20s} {'relative':>9s} {'multistep':>10s} {'all':>8s}")
    for name, res in results.items():
        cell = lambda cat: (lambda xs: f"{sum(x['hit'] for x in xs)}/{len(xs)}")([x for x in res if cat in (None, x["category"])])
        print(f"{name:20s} {cell('relative'):>9s} {cell('multistep'):>10s} {cell(None):>8s}")
    print(f"\nevery answer: {args.out}")


if __name__ == "__main__":
    main()
