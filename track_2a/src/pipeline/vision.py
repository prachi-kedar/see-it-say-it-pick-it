#!/usr/bin/env python3
"""Step 2: photo -> scene graph. Grounding DINO finds the objects, CLIP names them, OpenCV measures colour.

    python pipeline/vision.py --data dataset --out scene_graphs
    python pipeline/vision.py --data dataset --out scene_graphs --scene scene_06

For every scene folder with an rgb.png, writes:
    scene_graphs/<scene>.json   the objects, their colour, a pixel on each one, and which lies on which
    scene_graphs/<scene>.jpg    the photo with boxes and pixels drawn on it, to check by eye

Grounding DINO finds boxes well but often gives them the wrong name (it calls most things
"plastic part"), so it is only used to find boxes. CLIP looks at each box and says what it is
(gear, cube, mug, ...). Colour is measured from the pixels, which is more reliable than asking
either model for "red gear". The raw detector boxes are cached in <out>/<scene>.boxes.json, so
re-running after a change to naming or relations takes seconds (--no-cache to detect again). The pixel of each
object is the point deepest inside the part of its box that has the object's own colour, so it
lands in the middle of the object, not on its edge or on an object lying on top of it.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
from scene_io import load_scene  # noqa: E402

# Object types to look for. Key: what the detector is asked, one query at a time (asking for all
# of them in one prompt lowers every score); value: the noun used in object names.
VOCAB = {
    "gear": "gear", "cube": "cube", "mug": "mug", "tray": "tray", "plate": "plate", "saucer": "saucer",
    "mandarin": "mandarin", "banana": "banana", "plastic part": "part",
    # flat metal and white parts: "plastic part" misses the grey part half hidden under the cube in
    # scene_09, "metal bracket" finds it and the white parts too
    "metal bracket": "part",
}
# What CLIP chooses from. Two wordings per noun, averaged: each alone names about 28 of the 33
# ground-truth objects right, together 32 (the mistakes of the two differ). "table" = not an object.
NAMES = {
    "gear": ("a toothed gear wheel", "a photo of a toy gear with teeth, seen from above"),
    "cube": ("a plastic cube block", "a photo of a square plastic block, seen from above"),
    "mug": ("a coffee mug", "a photo of a mug, seen from above"),
    "mandarin": ("a mandarin orange fruit", "a photo of a mandarin, seen from above"),
    "banana": ("a banana", "a photo of a banana, seen from above"),
    "part": ("a pneumatic valve, an industrial component",
             "a photo of a small metal and plastic machine component, seen from above"),
    "plate": ("a round plate", "a photo of a plate, seen from above"),
    "tray": ("a plastic storage tray", "a photo of a storage tray, seen from above"),
    "table": ("an empty black table surface", "a photo of an empty dark table, seen from above"),
    # the plates have hearts printed on them, which the detector takes for parts
    "pattern": ("a printed pattern on a plate", "a photo of the decoration printed on a plate, seen from above"),
    # an operator's hand or wrist (scene_10): kept in the scene graph as a person, never picked
    "hand": ("a person's hand and wrist with a watch", "a photo of a human hand and arm, seen from above"),
}
NOT_OBJECTS = {"table", "pattern"}
MIN_SIDE_PX = 100     # the smallest object (the small white part) is ~170 px; thinner boxes are tray edges
ROUND_MAX_ASPECT = 1.3   # plates and saucers are round: a longer box is something else
CUBE_MAX_ASPECT = 1.25   # a cube seen from above is square, the parts are long
ROUND_MIN_PX = 350       # the saucer is ~500 px wide; a smaller "plate" is something else (a gear)
PATTERN_MAX_PX = 400     # a printed heart is ~200 px; a box this big called "pattern" is the plate itself
SHADES = {"white": "a white object", "grey": "a grey object", "black": "a black object"}
TRAY_MIN_SHARE = 0.15   # a box covering more than this share of the photo is a tray, whatever CLIP says
SAUCER_MAX_PX = 640   # a "plate" narrower than this is the saucer (the camera is always at the same height)
CONTAINERS = {"tray", "plate", "saucer"}   # places to put things; never picked
PEOPLE = {"hand"}                          # never picked, never touched
NO_COLOUR = {"mandarin", "banana", "plate", "saucer", "hand"}   # the noun already says what it is


def colour_names(rgb: np.ndarray) -> np.ndarray:
    """One colour word per pixel, from HSV (OpenCV hue is 0-179)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    out = np.full(h.shape, "grey", dtype=object)
    out[(s < 40) & (v > 170)] = "white"
    chroma = s >= 40
    out[chroma & ((h < 8) | (h >= 165))] = "red"
    out[chroma & (h >= 8) & (h < 20)] = "orange"
    out[chroma & (h >= 20) & (h < 35)] = "yellow"
    out[chroma & (h >= 35) & (h < 85)] = "green"
    out[chroma & (h >= 85) & (h < 130)] = "blue"
    out[chroma & (h >= 130) & (h < 165)] = "purple"
    out[v < 80] = "black"   # the black tray is ~52, the grey tray ~115
    return out


def iou(a, b) -> float:
    """Intersection over union of two boxes (x0, y0, x1, y1): 0 = apart, 1 = identical."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def overlap_of(a, b) -> float:
    """Share of box a that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0.0


class Detector:
    """Grounding DINO: finds boxes for a text query ("gear. cube. mug.") in one photo."""

    def __init__(self, model_id: str, device: str):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, query: str, box_threshold: float, text_threshold: float):
        """-> [(score, [x0, y0, x1, y1])] in pixels of rgb, for boxes above the thresholds."""
        text = query + "."
        inputs = self.processor(images=Image.fromarray(rgb), text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        kwargs = dict(text_threshold=text_threshold, target_sizes=[rgb.shape[:2]])
        try:   # the argument was renamed in newer transformers versions
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold, **kwargs)[0]
        except TypeError:
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold, **kwargs)[0]
        return [(float(score), [float(x) for x in box]) for score, box in zip(res["scores"].tolist(), res["boxes"].tolist())]

    def all_queries(self, rgb: np.ndarray, box_threshold: float, text_threshold: float, tiles: int = 2):
        """[(noun, score, box)] from one query per noun, on the whole photo and on tiles x tiles
        overlapping zoomed-in tiles (the detector shrinks the photo to ~800 px, so small objects
        are easier to find in a tile). A tile box that touches a cut edge of its tile is dropped:
        the object continues outside the tile and the whole photo or another tile has it whole."""
        h, w = rgb.shape[:2]
        windows = [(0, 0, w, h)]
        if tiles > 1:
            tw, th = int(w / tiles * 1.4), int(h / tiles * 1.4)   # 40% bigger, so tiles overlap
            for i in range(tiles):
                for j in range(tiles):
                    x0, y0 = min(w - tw, round(i * (w - tw) / (tiles - 1))), min(h - th, round(j * (h - th) / (tiles - 1)))
                    windows.append((x0, y0, x0 + tw, y0 + th))
        out, margin = [], 8
        for x0, y0, x1, y1 in windows:
            crop = rgb[y0:y1, x0:x1]
            for query, noun in VOCAB.items():
                for score, (bx0, by0, bx1, by1) in self(crop, query, box_threshold, text_threshold):
                    cut = ((x0 > 0 and bx0 < margin) or (y0 > 0 and by0 < margin)
                           or (x1 < w and bx1 > crop.shape[1] - margin) or (y1 < h and by1 > crop.shape[0] - margin))
                    if not cut:
                        out.append((noun, score, [bx0 + x0, by0 + y0, bx1 + x0, by1 + y0]))
        return out


class Namer:
    """CLIP: crop of one box -> noun from NAMES."""

    def __init__(self, model_id: str, device: str):
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.nouns = list(NAMES)

    @torch.no_grad()
    def _probs(self, crop: np.ndarray, texts: list[str]):
        inputs = self.processor(text=texts, images=Image.fromarray(crop), return_tensors="pt", padding=True).to(self.device)
        return self.model(**inputs).logits_per_image.softmax(-1)[0]

    def ranked(self, crop: np.ndarray) -> list[tuple[str, float]]:
        """[(noun, probability)], most likely first."""
        probs = sum(self._probs(crop, [NAMES[n][i] for n in self.nouns]) for i in range(2)) / 2
        return sorted(zip(self.nouns, probs.tolist()), key=lambda t: -t[1])

    def __call__(self, crop: np.ndarray) -> tuple[str, float]:
        return self.ranked(crop)[0]

    def shade(self, crop: np.ndarray) -> str:
        """White, grey or black, for objects without hue. Pixel brightness cannot tell: in these photos
        the white part (median brightness 170) is barely brighter than a grey one (162)."""
        words = list(SHADES)
        return words[int(self._probs(crop, [SHADES[w] for w in words]).argmax())]


def shape_check(ranked: list[tuple[str, float]], aspect: float, big: bool, round_size: bool) -> tuple[str, float]:
    """CLIP's best noun, corrected by the shape of the box where CLIP is known to confuse two nouns:
    a long or small box is not a plate or saucer (the flat grey part was named "plate"), and between cube and
    part (CLIP gives both about the same probability) a square box is the cube."""
    noun, p = ranked[0]
    if noun == "pattern" and big:   # a whole decorated plate or saucer, not a print on one
        noun, p = next((n, q) for n, q in ranked if n not in NOT_OBJECTS)
    if noun in ("plate", "saucer") and (aspect > ROUND_MAX_ASPECT or not round_size):
        noun, p = next((n, q) for n, q in ranked if n not in CONTAINERS and n not in NOT_OBJECTS)
    if {noun, ranked[1][0]} == {"cube", "part"} or (noun in ("cube", "part") and ranked[0][0] in CONTAINERS):
        noun = "cube" if aspect <= CUBE_MAX_ASPECT else "part"
    return noun, p


def point_on_object(colours: np.ndarray, box, colour: str | None, exclude=()):
    """Pixel deepest inside the object's own colour within its box (box centre as fallback).
    exclude: boxes of objects lying on this one, which can have the same colour."""
    x0, y0, x1, y1 = (int(round(c)) for c in box)
    centre = [(x0 + x1) // 2, (y0 + y1) // 2]
    patch = colours[y0:y1, x0:x1]
    mask = (patch == colour) if colour else (patch != "black")
    for ex0, ey0, ex1, ey1 in exclude:
        mask[max(0, ey0 - y0):max(0, ey1 - y0), max(0, ex0 - x0):max(0, ex1 - x0)] = False
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if mask.sum() < 50:
        return centre
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    padded = cv2.copyMakeBorder((lab == biggest).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [x0 + int(u) - 1, y0 + int(v) - 1]


def stacking(objs: list[dict], colours: np.ndarray, min_overlap: float = 0.15, min_share: float = 0.3):
    """Sets o["on_top_of"] = [ids of the objects o lies on]. Seen from above, two objects whose
    outlines overlap lie on each other, and the one on top is the one visible where they overlap:
    - different colours: whichever colour fills more of the overlap of the two boxes is on top
      (a box corner that only covers table has little of either colour: not touching);
    - same colour (a red cube on red gears): a box mostly inside a bigger one is on top, and an
      object overlapping two others that do not overlap each other bridges them, so it is on top.
    Containers are left out: a saucer edge over a part is not something the gripper must move."""
    pickable = [o for o in objs if o["noun"] not in CONTAINERS | PEOPLE]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])

    def inter(a, b):
        return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]

    def overlaps(a, b) -> bool:
        i = inter(a["box"], b["box"])
        return i[2] > i[0] and i[3] > i[1] and area(i) >= min_overlap * min(area(a["box"]), area(b["box"]))

    for o in objs:
        o["on_top_of"] = []
    for i, a in enumerate(pickable):
        for b in pickable[i + 1:]:
            if not overlaps(a, b):
                continue
            if a["colour"] != b["colour"]:
                x0, y0, x1, y1 = inter(a["box"], b["box"])
                patch = colours[y0:y1, x0:x1]
                sa, sb = float((patch == a["colour"]).mean()), float((patch == b["colour"]).mean())
                if max(sa, sb) >= min_share:
                    top, bottom = (a, b) if sa > sb else (b, a)
                    top["on_top_of"].append(bottom["id"])
            else:
                small, big = (a, b) if area(a["box"]) < area(b["box"]) else (b, a)
                if overlap_of(small["box"], big["box"]) > 0.6:
                    small["on_top_of"].append(big["id"])
    for o in pickable:   # same colour, bridging two objects
        same = [b for b in pickable if b is not o and b["colour"] == o["colour"] and overlaps(o, b)
                and b["id"] not in o["on_top_of"] and o["id"] not in b["on_top_of"]]
        if any(not overlaps(p, q) for k, p in enumerate(same) for q in same[k + 1:]):
            o["on_top_of"] += [b["id"] for b in same]
    for o in pickable:   # the pixel of an object below must not land on a same-coloured object on top
        tops = [t["box"] for t in pickable if o["id"] in t["on_top_of"] and t["colour"] == o["colour"]]
        if tops:
            o["pixel"] = point_on_object(colours, o["box"], None if o["noun"] in NO_COLOUR else o["colour"], tops)


def build_scene_graph(rgb: np.ndarray, detections, name, min_score: float = 0.25,
                      max_box_share: float = 0.6, same_object_iou: float = 0.7) -> list[dict]:
    """detections: [(query noun, score, box)] from the detector; name: crop -> (noun, confidence).
    Boxes from all queries are grouped per object; the detector score only decides what is an object."""
    h, w = rgb.shape[:2]
    colours = colour_names(rgb)

    # 1. group boxes that cover the same object; the group keeps its best box and best score per noun
    groups = []
    for noun, score, box in sorted(detections, key=lambda d: -d[1]):
        box = [max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3])]
        if (box[2] - box[0]) * (box[3] - box[1]) > max_box_share * w * h:
            continue
        g = next((g for g in groups if iou(g["box"], box) > same_object_iou), None)
        if g is None:
            groups.append({"box": box, "scores": {noun: score}})
        else:
            g["scores"][noun] = max(g["scores"].get(noun, 0.0), score)
    kept = []
    for g in sorted(groups, key=lambda g: -max(g["scores"].values())):
        score = max(g["scores"].values())
        if score < min_score:
            continue
        box = [round(c) for c in g["box"]]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if min(bw, bh) < MIN_SIDE_PX:
            continue
        ranked = name.ranked(rgb[box[1]:box[3], box[0]:box[2]])
        noun, confidence = shape_check(ranked, max(bw, bh) / min(bw, bh), max(bw, bh) > PATTERN_MAX_PX,
                                        min(bw, bh) >= ROUND_MIN_PX)
        if bw * bh > TRAY_MIN_SHARE * w * h:
            noun = "tray"
        elif noun in NOT_OBJECTS:
            continue
        if noun == "plate" and max(box[2] - box[0], box[3] - box[1]) < SAUCER_MAX_PX:
            noun = "saucer"
        # a second box on the same object, with the same name: keep the first (higher detector score)
        if any(k["noun"] == noun and iou(k["box"], box) > 0.5 for k in kept):
            continue
        kept.append({"noun": noun, "score": round(score, 3), "name_confidence": round(confidence, 3), "box": box})

    # 2. colour = most common colour word in the box, ignoring the black table (except for trays)
    for o in kept:
        x0, y0, x1, y1 = o["box"]
        words, counts = np.unique(colours[y0:y1, x0:x1], return_counts=True)
        tally = dict(zip(words, counts))
        if o["noun"] != "tray":
            tally.pop("black", None)
        o["colour"] = max(tally, key=tally.get) if tally else None
        if o["colour"] in SHADES and o["noun"] not in NO_COLOUR and o["noun"] != "tray":
            o["colour"] = name.shade(rgb[y0:y1, x0:x1])
        if o["noun"] in CONTAINERS:   # only a place to put things: the box centre is enough
            o["pixel"] = [(x0 + x1) // 2, (y0 + y1) // 2]
        else:
            o["pixel"] = point_on_object(colours, o["box"], None if o["noun"] in NO_COLOUR else o["colour"])

    # 2a. the workcell has a grey tray on the left and a black tray on the right (docs/DATA.md). The
    # black tray fills most of the right half, so the detector often skips it: take the largest black area.
    if not any(o["noun"] == "tray" and o["colour"] == "black" for o in kept):
        black = (colours == "black").astype(np.uint8)
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(black)
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            if bw * bh > TRAY_MIN_SHARE * w * h:
                kept.append({"noun": "tray", "colour": "black", "score": 0.0, "name_confidence": 0.0,
                             "box": [x, y, x + bw, y + bh], "pixel": [x + bw // 2, y + bh // 2]})

    # 2b. a "plate" or "saucer" lying inside another one is its printed decoration
    for o in [o for o in kept if o["noun"] in ("plate", "saucer")]:
        if any(c is not o and c["noun"] in ("plate", "saucer") and overlap_of(o["box"], c["box"]) > 0.9
               and (c["box"][2] - c["box"][0]) > (o["box"][2] - o["box"][0]) for c in kept):
            kept.remove(o)

    # 2c. one object seen as two boxes (a mug's rim and the whole mug): same noun and colour, one box
    # inside the other. Objects lying on each other differ in colour or noun, so they are kept.
    # A mug never lies in a mug: a small "mug" inside a mug's box is its handle, whatever its colour.
    for o in list(kept):
        if any(k is not o and k in kept and k["noun"] == o["noun"]
               and (k["colour"] == o["colour"] and k["score"] >= o["score"]
                    and max(overlap_of(o["box"], k["box"]), overlap_of(k["box"], o["box"])) > 0.8
                    or o["noun"] == "mug" and overlap_of(o["box"], k["box"]) > 0.8)
               for k in kept):
            kept.remove(o)

    # 3. names: colour_noun, numbered left to right when there are several (red_gear_1, red_gear_2)
    for o in kept:
        o["base"] = o["noun"] if o["noun"] in NO_COLOUR or not o["colour"] else f"{o['colour']}_{o['noun']}"
    for base in {o["base"] for o in kept}:
        same = sorted((o for o in kept if o["base"] == base), key=lambda o: o["pixel"][0])
        for i, o in enumerate(same, 1):
            o["id"] = base if len(same) == 1 else f"{base}_{i}"

    # 4. relations: inside a container, and which objects each one lies on (see stacking)
    for o in kept:
        o["inside"] = None
        if o["noun"] in CONTAINERS:
            continue
        u, v = o["pixel"]
        holders = [c for c in kept if c["noun"] in CONTAINERS and c["box"][0] <= u <= c["box"][2] and c["box"][1] <= v <= c["box"][3]]
        if holders:   # the smallest: a mandarin on a plate on the black tray is on the plate
            o["inside"] = min(holders, key=lambda c: (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]))["id"]
    stacking(kept, colours)

    keys = ("id", "noun", "colour", "pixel", "box", "score", "name_confidence", "inside", "on_top_of")
    return [{k: o[k] for k in keys} for o in sorted(kept, key=lambda o: o["id"])]


def draw(rgb: np.ndarray, objects: list[dict]) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for o in objects:
        x0, y0, x1, y1 = o["box"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 4)
        cv2.circle(img, tuple(o["pixel"]), 14, (255, 0, 255), -1)
        cv2.putText(img, o["id"], (x0 + 6, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--out", type=Path, default=Path("outputs/scene_graphs"))
    ap.add_argument("--scene", help="only this scene, e.g. scene_06")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--no-cache", action="store_true", help="run the detector again even if boxes are cached")
    ap.add_argument("--box-threshold", type=float, default=0.2, help="keep detector boxes above this")
    ap.add_argument("--min-score", type=float, default=0.25, help="keep objects whose best noun scores above this")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} and {args.clip_model} on {device}")
    detect = Detector(args.model, device)
    name = Namer(args.clip_model, device)
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / "rgb.png").exists())
    if args.scene:
        scenes = [p for p in scenes if p.name == args.scene]
    for d in scenes:
        rgb = load_scene(d).rgb
        cache = args.out / f"{d.name}.boxes.json"
        if cache.exists() and not args.no_cache:
            detections = json.loads(cache.read_text())
        else:
            detections = detect.all_queries(rgb, args.box_threshold, args.box_threshold)
            cache.write_text(json.dumps(detections))
        objects = build_scene_graph(rgb, detections, name, args.min_score)
        graph = {"scene": d.name, "image_size": [rgb.shape[1], rgb.shape[0]], "objects": objects}
        (args.out / f"{d.name}.json").write_text(json.dumps(graph, indent=2))
        cv2.imwrite(str(args.out / f"{d.name}.jpg"), draw(rgb, objects))
        print(f"{d.name}: " + ", ".join(o["id"] for o in objects))


if __name__ == "__main__":
    main()
