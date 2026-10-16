#!/usr/bin/env python3
"""Step 3: scene graph + instruction -> answer (plan, question or refusal).

    python pipeline/plan.py --data dataset --graphs scene_graphs --out results.json
    python starter/evaluate.py results.json --data dataset

The work is split by what each side does well:

  Apertus (language)  reads the instruction and writes down what it asks for, in a fixed structure:
                      the motion (pick and place, throw, push, touch a person), which object
                      ("the red gear closest to the mandarin", "everything in the black tray except
                      the mug"), where it goes, and what must not move. Constrained decoding: Ollama
                      gets a JSON schema, so the reply always parses and every field is from a fixed list.
  code (geometry)     finds those objects in the scene graph from step 2: colours, distances,
                      left/right, which object lies on which. Then it builds the plan (an object lying
                      on another one is moved first), or asks when several objects fit and nothing
                      says which, or refuses with the reason.

An 8B model is reliable at the first and unreliable at the second (with the scene graph as text it
picked the mug for "the object closest to the mug"), so it is not asked to do the second.

Safety is checked twice: Apertus classifies the motion, and a short list of words (throw, push,
"my hand", ...) refuses on its own, so one mistake of the model does not reach the robot.
Every raw reply and every decision goes to <out>.raw.jsonl.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
from scene_io import load_scene  # noqa: E402

MODEL = os.environ.get("LLM_NAME", "hf.co/Colby/apertus-v1.5-8b-text-Q4_K_M-GGUF")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
CONTAINERS = {"tray", "plate", "saucer"}
PEOPLE = {"hand"}
OBJECT_NOUNS = ["gear", "cube", "mug", "mandarin", "banana", "part"]   # what the gripper can pick
NOUNS = OBJECT_NOUNS + ["plate", "saucer", "tray"]
GROUPS = {"fruit": {"mandarin", "banana"}, "object": set(OBJECT_NOUNS)}
CONFUSED = {"cube": "part", "part": "cube"}   # the vision step mixes these up (see vision.shape_check)
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "white", "grey", "black"]
SHADES = {"white", "grey", "black"}   # the camera cannot tell these apart reliably (see vision.py)
GRIPPER_MM = 85
REL_WORDS = {"closest_to": "close to", "farthest_from": "far from", "left_of": "to the left of",
             "right_of": "to the right of", "on_top_of": "lying on", "under": "under", "in": "in"}

MOTIONS = {   # motion -> refusal reason (None: safe)
    "pick_up": None,
    "move_or_put": None,
    "throw_or_drop": "Throwing or dropping makes the object move without control, which is not safe. "
                     "I only pick objects up and set them down.",
    "push_or_pull": "Pushing or pulling an object off or out from under another one makes objects slide "
                    "or fall without control, which is not safe. I only pick objects up and set them down.",
    "involves_person": "The robot must not move to a person or take things from or give things into a "
                       "person's hand.",
}
# A second, independent safety net: these words refuse whatever Apertus says.
UNSAFE_WORDS = [
    (r"\b(throw|toss|fling|hurl|drop|knock|kick|smash|shake|swing|flick)\w*\b", "throw_or_drop"),
    (r"\b(push|shove|slide)\w*\b", "push_or_pull"),
    (r"\b(pull|yank|drag|jerk)\w*\b.*\b(out|under|underneath|from)\b", "push_or_pull"),
    (r"\b(my|your|his|her|their|someone'?s|a person'?s|the person'?s|operator'?s)\s+(hands?|arms?|fingers?|head|face|lap)\b",
     "involves_person"),
    (r"\bhand\s+(it|this|that|them)?\s*(to|me|over)\b", "involves_person"),
    (r"\b(person|human|operator|somebody|someone|colleague)\b", "involves_person"),
]

PARSER = f"""You turn one instruction for a robot arm into a fixed structure. The arm picks objects
from a table and places them in a tray, on a plate, on a saucer or on the table. You do not see the
table and you do not decide whether the instruction can be done: you only write down what it asks
for. Always write at least one task, also for an object that may not be there.

Fields:
- "motion": "pick_up" for picking up, taking, grabbing. "move_or_put" for moving, putting, placing,
  transferring, clearing (also "move everything from A to B"). "throw_or_drop" for throwing, tossing, dropping. "push_or_pull" for pushing, sliding, knocking off,
  pulling an object out from under another one. "involves_person" when the robot would touch a person,
  take something out of someone's hand or put something into a hand.
- "tasks": one task per object (or group of objects) to move, in the order of the instruction.
  "put A on the plate and B in the tray" is two tasks.
  - "target": the object to pick, described as in the instruction:
    - "noun": one of {NOUNS + ["fruit", "object"]}; "object" for "object", "thing", "everything",
      "all"; "none" when no object is named ("that one", "it" with nothing before it);
      "other" for an object type not in the list.
    - "colour": the colour word in front of it, or "any".
    - "size": "small" or "large" if the instruction says so, else "any".
    - "relation" with "ref_noun"/"ref_colour" (the other object it is described by):
      "closest_to" (closest to, nearest, next to, beside), "farthest_from", "left_of", "right_of",
      "on_top_of" (lying on, on top of, sitting on), "under" (under, below, underneath),
      "in" (in, from, inside a tray or on a plate or saucer), "nothing_on_it" (with nothing on it,
      free), else "none".
    - "position": "leftmost" ("on the left", "the left one"), "rightmost", "second_from_left",
      "second_from_right", "topmost" ("at the top"), "bottommost", "middle", else "none".
    - "words": the words of the instruction that name the target (write them first).
  - "all": true for "everything", "all", "both", "all the gears"; false for one object.
  - "except": objects excluded with "except" or "but not" (noun and colour).
  - "to_noun"/"to_colour": where it is put: "tray", "plate", "saucer", "table", or another noun if
    the instruction says so (for example "in the mug"); "none" if it is only picked up.
- "keep_still": objects that must not move: "without moving X", "leave X where it is", "don't touch X".
  "it" means the object named before it.

Write "analysis" first: one short sentence on what the instruction asks.

Examples:
"pick up the green cube" -> motion pick_up; one task: target cube, colour green; to none
"pick up the purple banana" -> target banana, colour purple (whether it exists is checked later)
"put the small white part in the black tray" -> target part, colour white, size small; to tray black
"pick up the gear lying on the cube" -> target gear, relation on_top_of, ref cube
"pick up the cube with nothing on it" -> target cube, relation nothing_on_it
"put the blue part under the mug on the plate" -> target part blue, relation under, ref mug; to plate
"pick up the cube nearest to the banana" -> target cube, relation closest_to, ref banana
"pick up the third gear from the left" -> not in the list of positions: position none
"move all objects from the black tray to the grey tray except the mug" -> motion move_or_put; target object, relation in,
  ref tray black, all true, except [mug]; to tray grey
"put the cube on the plate and the gear in the black tray" -> two tasks
"pick up the mug without moving the gear" -> target mug; keep_still [gear]
"put the banana in the tray and leave it there" -> target banana; to tray; keep_still [banana]
"toss the gear into the tray" -> motion throw_or_drop
"slide the cube out from under the mug" -> motion push_or_pull
"put the cube in my hand" -> motion involves_person
"give me that one" -> target noun none
"""


# ----------------------------------------------------------------------------- Apertus
# Ollama writes the keys of a JSON object in alphabetical order, whatever order the schema gives, and
# the model decides each field as it writes it. The letter prefixes make that order the order of
# thought (the words first, the noun before its relation, the relation before its reference);
# plain() takes them off again.
def obj(fields: dict) -> dict:
    props = {f"{chr(97 + i)}_{k}": v for i, (k, v) in enumerate(fields.items())}
    return {"type": "object", "properties": props, "required": list(props)}


def desc_schema(simple: bool = False) -> dict:
    fields = {"words": {"type": "string"},
              "colour": {"enum": COLOURS + ["any"]},
              "noun": {"enum": NOUNS + ["fruit", "object", "none", "other"]}}
    if not simple:
        fields |= {"size": {"enum": ["any", "small", "large"]},
                   "relation": {"enum": ["none", "closest_to", "farthest_from", "left_of", "right_of",
                                         "on_top_of", "under", "in", "nothing_on_it"]},
                   "ref_colour": {"enum": COLOURS + ["any"]},
                   "ref_noun": {"enum": NOUNS + ["fruit", "object", "none"]},
                   "position": {"enum": ["none", "leftmost", "rightmost", "second_from_left", "second_from_right",
                                         "topmost", "bottommost", "middle"]}}
    return obj(fields)


def parser_schema() -> dict:
    task = obj({"target": desc_schema(),
                "all": {"type": "boolean"},
                "except": {"type": "array", "items": desc_schema(simple=True)},
                "to_colour": {"enum": COLOURS + ["any"]},
                "to_noun": {"enum": NOUNS + ["table", "none", "other"]}})
    return obj({"analysis": {"type": "string"},
                "motion": {"enum": list(MOTIONS)},
                "tasks": {"type": "array", "items": task, "minItems": 1},
                "keep_still": {"type": "array", "items": desc_schema(simple=True)}})


def plain(x):
    """Takes the letter prefixes off the keys of a parsed reply: {"a_words": ..} -> {"words": ..}."""
    if isinstance(x, dict):
        return {re.sub(r"^[a-z]_", "", k): plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [plain(v) for v in x]
    return x


def ask_apertus(base_url: str, model: str, system: str, user: str, schema: dict, timeout: float) -> str:
    body = {"model": model, "stream": False, "format": schema, "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def unload(base_url: str, model: str):
    """Free the GPU (the vision models need it)."""
    body = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(base_url.rstrip("/") + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except (urllib.error.URLError, TimeoutError):
        pass


# ----------------------------------------------------------------------------- the scene
class NotDone(Exception):
    """The instruction cannot become a plan: kind is "refuse" or "ask_user", text is the reason or question."""

    def __init__(self, kind: str, text: str):
        super().__init__(text)
        self.kind, self.text = kind, text


def name_of(o: dict) -> str:
    """'the red gear', 'the mandarin'"""
    plain = o["noun"] in ("mandarin", "banana", "plate", "saucer", "hand") or not o.get("colour")
    return "the " + (o["noun"] if plain else f"{o['colour']} {o['noun']}")


def said(d: dict) -> str:
    """How a description reads: 'red gear', 'object'."""
    noun = d.get("words") or d["noun"]
    colour = d.get("colour", "any")
    return noun if colour == "any" or colour in noun else f"{colour} {noun}"


class Scene:
    def __init__(self, graph: dict):
        self.objs = graph["objects"]
        self.by_id = {o["id"]: o for o in self.objs}
        for o in self.objs:   # older scene graphs: on_top_of was one id or None
            if not isinstance(o.get("on_top_of"), list):
                o["on_top_of"] = [o["on_top_of"]] if o.get("on_top_of") else []
        self.pickable = [o for o in self.objs if o["noun"] not in CONTAINERS | PEOPLE]

    def tops(self, o: dict) -> list[dict]:
        """Objects lying on o."""
        return [p for p in self.pickable if o["id"] in p["on_top_of"]]

    @staticmethod
    def within(o: dict, c: dict) -> bool:
        u, v = o["pixel"]
        x0, y0, x1, y1 = c["box"]
        return o is not c and x0 <= u <= x1 and y0 <= v <= y1

    @staticmethod
    def gap(a: dict, b: dict) -> float:
        """Distance between the edges of two objects, taken as round: centre to centre minus both radii.
        (Box corners of round gears overlap long before the gears do, so the gap between boxes misleads;
        centres alone favour small objects next to a big plate.)"""
        centre = lambda o: ((o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2)
        radius = lambda o: (o["box"][2] - o["box"][0] + o["box"][3] - o["box"][1]) / 4
        return math.dist(centre(a), centre(b)) - radius(a) - radius(b)

    def match(self, noun: str, colour: str, pool: list[dict] | None = None) -> list[dict]:
        """Objects a noun and colour can mean. Exact colour first; white/grey/black stand in for each
        other (except on trays) and cube/part for each other only when nothing fits exactly."""
        pool = self.objs if pool is None else pool
        nouns = GROUPS.get(noun, {noun})
        same = [o for o in pool if o["noun"] in nouns]
        if not same and noun in CONFUSED:
            same = [o for o in pool if o["noun"] == CONFUSED[noun]]
        if colour == "any":
            return same
        exact = [o for o in same if o.get("colour") == colour]
        if exact:
            return exact
        return [o for o in same if colour in SHADES and o.get("colour") in SHADES and o["noun"] != "tray"]

    def where(self, o: dict, among: list[dict]) -> str:
        """Words that tell o apart from the others: 'the one on the left'."""
        xs = sorted(among, key=lambda p: p["pixel"][0])
        i = xs.index(o)
        if len(xs) == 2:
            return "the one on the left" if i == 0 else "the one on the right"
        return "the one on the left" if i == 0 else "the one on the right" if i == len(xs) - 1 \
            else "the one in the middle" if len(xs) == 3 else f"the {ordinal(i + 1)} from the left"

    # ------------------------------------------------------------------ one description -> objects
    def resolve(self, d: dict, all_of_them: bool, exclude: list[dict]) -> list[dict]:
        noun, colour, word = d["noun"], d["colour"], said(d)
        if noun == "none":
            raise NotDone("ask_user", "Which object do you mean? Please name it, for example \"the red cube\".")
        if noun == "other":
            raise NotDone("refuse", f"I do not see a {word} on the table.")
        if noun in PEOPLE:
            raise NotDone("refuse", MOTIONS["involves_person"])
        pool = self.objs if noun in CONTAINERS else self.pickable
        found = self.match(noun, colour, pool)
        if not found:
            raise NotDone("refuse", f"There is no {word} on the table.")
        if noun in CONTAINERS:
            c = found[0]
            if c["noun"] == "tray":
                raise NotDone("refuse", f"{name_of(c).capitalize()} is a place to put things; it cannot be picked.")
            raise NotDone("refuse", f"{name_of(c).capitalize()} is wider than the {GRIPPER_MM} mm gripper "
                                    "opens, so it cannot be picked.")
        found = [o for o in found if o not in exclude]

        rel = d.get("relation", "none")
        if rel not in ("none", "nothing_on_it"):
            ref_noun = d.get("ref_noun", "none")
            if ref_noun == "none":
                ref_noun = "object"
            refs = self.match(ref_noun, d.get("ref_colour", "any"),
                              self.objs if ref_noun in CONTAINERS else self.pickable)
            ref_word = said({"noun": ref_noun, "colour": d.get("ref_colour", "any")})
            if not refs:
                raise NotDone("refuse", f"There is no {ref_word} on the table.")
            found = self.related(found, rel, refs)
            if not found:
                raise NotDone("refuse", f"No {word} is {REL_WORDS[rel]} the {ref_word}.")
        elif rel == "nothing_on_it":
            found = [o for o in found if not self.tops(o)]
            if not found:
                raise NotDone("refuse", f"Every {word} has something lying on it.")

        pos = d.get("position", "none")
        if pos != "none" and len(found) > 1:
            vertical = pos in ("topmost", "bottommost", "second_from_top", "second_from_bottom")
            xs = sorted(found, key=(lambda o: o["pixel"][1]) if vertical else (lambda o: o["pixel"][0]))
            i = {"leftmost": 0, "topmost": 0, "rightmost": -1, "bottommost": -1, "second_from_left": 1,
                 "second_from_top": 1, "second_from_right": -2, "second_from_bottom": -2,
                 "middle": len(xs) // 2}[pos]
            if pos.startswith("second") and len(xs) < 2 or pos == "middle" and len(xs) % 2 == 0:
                raise NotDone("ask_user", f"Which {word} do you mean? I see {len(xs)}.")
            found = [xs[i]]
        size = d.get("size", "any")
        if size != "any" and len(found) > 1:
            area = lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])
            xs = sorted(found, key=area)
            if area(xs[-1]) > 1.5 * area(xs[0]):
                found = [xs[0] if size == "small" else xs[-1]]

        if len(found) > 1 and not all_of_them:
            places = ", ".join(self.where(o, found) for o in found[:-1]) + " or " + self.where(found[-1], found)
            raise NotDone("ask_user", f"I see {len(found)} objects that fit \"{word}\". Which one do you mean: {places}?")
        return found

    def related(self, found: list[dict], rel: str, refs: list[dict]) -> list[dict]:
        ids = {r["id"] for r in refs}
        if rel == "on_top_of":
            return [o for o in found if set(o["on_top_of"]) & (ids - {o["id"]})]
        if rel == "under":
            return [o for o in found if any(o["id"] in r["on_top_of"] for r in refs if r is not o)]
        if rel == "in":
            return [o for o in found if any(self.within(o, r) for r in refs)]
        others = [o for o in found if o["id"] not in ids]
        if not others:
            return []
        dist = lambda o: min(self.gap(o, r) for r in refs)
        if rel == "closest_to":
            return [min(others, key=dist)]
        if rel == "farthest_from":
            return [max(others, key=dist)]
        if rel in ("left_of", "right_of"):
            sign = -1 if rel == "left_of" else 1
            side = [o for o in others if any(sign * (o["pixel"][0] - r["pixel"][0]) > 0 for r in refs)]
            # "to the left of" means more sideways than up or down; if nothing is, any object on that side
            level = [o for o in side if any(sign * (o["pixel"][0] - r["pixel"][0]) >= abs(o["pixel"][1] - r["pixel"][1])
                                            for r in refs)]
            return [min(level or side, key=dist)] if side else []
        return found

    def destination(self, task: dict) -> str | None:
        noun, colour = task["to_noun"], task["to_colour"]
        if noun == "none":
            return None
        if noun == "table":
            return "table"
        if noun not in CONTAINERS:
            raise NotDone("refuse", f"I can only put objects in a tray, on the plate, on the saucer or on the "
                                    f"table, not in or on the {noun if noun != 'other' else 'object you named'}.")
        found = self.match(noun, colour)
        word = noun if colour == "any" else f"{colour} {noun}"
        if not found:
            raise NotDone("refuse", f"There is no {word} on the table to put it in.")
        if len(found) > 1:
            raise NotDone("ask_user", f"There are {len(found)} {noun}s. In which {noun} should I put it, "
                                      f"{self.where(found[0], found)} or {self.where(found[-1], found)}?")
        return found[0]["id"]


def ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth"}.get(n, f"{n}th")


# ----------------------------------------------------------------------------- plan
def build_plan(scene: Scene, jobs: list[tuple[list[dict], str | None]]) -> list[dict]:
    """jobs: [(objects, destination id or None)]. The gripper holds one object at a time; anything lying
    on an object is moved first: with the same destination if it is to be moved anyway, else aside
    onto the table."""
    actions, moved = [], set()
    to_move = {o["id"]: dest for objs, dest in jobs for o in objs}

    def move(o: dict, dest: str | None, chain: tuple = ()):
        if o["id"] in moved or o["id"] in chain:
            return
        for top in scene.tops(o):
            move(top, to_move.get(top["id"], "table"), chain + (o["id"],))
        actions.append({"action": "pick", "object": o["id"]})
        if dest:
            actions.append({"action": "place", "object": dest})
        moved.add(o["id"])

    for objs, dest in jobs:
        for o in objs:
            move(o, dest)
    return actions


# ----------------------------------------------------------------------------- checking the parse
# Apertus sometimes fills a field the instruction says nothing about (a tray to put a picked-up object
# in, "left_of" next to "leftmost"). Every relation, position, destination and "all" must have its
# words in the instruction; what has none is dropped.
EVIDENCE = {
    "closest_to": r"\b(clos|near|next|beside|by)", "farthest_from": r"\b(far|furthest)",
    "left_of": r"\bleft\s+of\b", "right_of": r"\bright\s+of\b",
    "on_top_of": r"\b(on|on top|top of|lying|lies|sitting|resting|onto)\b", "under": r"\b(under|below|beneath)",
    "in": r"\b(in|inside|from|on|within)\b", "nothing_on_it": r"\b(nothing|free|empty|uncovered)\b",
    "leftmost": r"\bleft", "rightmost": r"\bright", "second_from_left": r"\bsecond\b.*\bleft",
    "second_from_right": r"\bsecond\b.*\bright", "topmost": r"\btop", "bottommost": r"\bbottom",
    "second_from_top": r"\bsecond\b.*\btop", "second_from_bottom": r"\bsecond\b.*\bbottom",
    "middle": r"\b(middle|centre|center)\b",
}
ALL_WORDS = r"\b(all|every|everything|both|each|two|three|four)\b"
# Relations that compare with another object: with no object named to compare with ("closest to the
# bottom of the image") the words are a position, not a relation.
NEEDS_REFERENCE = {"closest_to", "farthest_from", "left_of", "right_of"}
NOT_COLOUR = rf"\b(?:is\s+not|isn'?t|are\s+not|aren'?t|non-?)\s*({'|'.join(COLOURS + ['gray'])})\b"
PLACE_WORDS = r"\b(in|into|on|onto|to|inside|back)\b"
PUT_VERBS = r"\b(put|place|move|set|transfer|bring|stack|clear|return|sort|leave|lay)\w*\b"
PRONOUNS = {"it", "that", "this", "that one", "this one", "them", "one", "those", "these"}


def noun_said(noun: str, text: str) -> bool:
    words = {"object": r"\b(object|thing|everything|item|all|both|each)", "fruit": r"\bfruit",
             "table": r"\b(table|aside|down)\b"}.get(noun, rf"\b{noun}")
    return bool(re.search(words, text))


def colour_said(colour: str, noun: str, text: str) -> bool:
    """Is the colour written in front of this noun ("the black tray", "the black small mug")? Apertus
    sometimes moves a colour to another noun of the sentence ("the mug in the black tray" -> black mug)."""
    if colour == "any" or noun not in NOUNS:
        return True
    c = "(grey|gray)" if colour == "grey" else colour
    return bool(re.search(rf"\b{c}\s+(\w+\s+){{0,2}}{noun}", text))


def repair(parsed: dict, instruction: str) -> tuple[dict, list[str]]:
    """Checks Apertus's parse against the words of the instruction and returns (fixed parse, notes).

    Rules, none written for a particular instruction:
      - a relation, position, destination or "all" needs its words in the instruction, else it is dropped;
      - a relation that compares with another object needs that object ("closest to the bottom of the
        image" is a position, not a relation); "top" alone is a position, not "lying on";
      - a colour counts only in front of the noun it is written with;
      - "two/three/four/both/all" mean all, and the same task twice is one task;
      - "second from the top/bottom" and "is not <colour>" are added here, because the schema of the
        reply has no field for them (the Apertus cache stays valid);
      - a pick-up has no destination; "it"/"that one" name the object before them, or none.
    Every change is written to notes, which go to the .raw.jsonl log."""
    text, notes = instruction.lower(), []
    tasks = parsed.get("tasks") or []
    for t in tasks:
        d = t["target"]
        if (d.get("words") or "").strip().lower() in PRONOUNS:
            d["noun"] = "none"   # "that one" names no object, whatever noun was chosen
        rel = d.get("relation", "none")
        if rel != "none" and not re.search(EVIDENCE[rel], text):
            notes.append(f"dropped relation {rel}")
            d["relation"] = "none"
        if d["relation"] not in ("none", "nothing_on_it") and not noun_said(d.get("ref_noun", "none"), text) \
                and d.get("ref_noun") not in ("none", "object"):
            notes.append(f"dropped relation {rel}: no {d['ref_noun']} in the instruction")
            d["relation"] = "none"
        if d["relation"] in NEEDS_REFERENCE and d.get("ref_noun", "none") == "none":
            notes.append(f"dropped relation {d['relation']}: nothing to compare with")
            d["relation"] = "none"
        if d["relation"] == "none" and re.search(EVIDENCE["nothing_on_it"], text) and re.search(r"\bon\b", text):
            d["relation"] = "nothing_on_it"
            notes.append("added nothing_on_it")
        pos = d.get("position", "none")
        if pos != "none" and not re.search(EVIDENCE[pos], text):
            notes.append(f"dropped position {pos}")
            d["position"] = "none"
        if d["position"] != "none" and d["relation"] in ("left_of", "right_of", "farthest_from", "closest_to") \
                and not re.search(EVIDENCE[d["relation"]], text.replace("second", "")):
            d["relation"] = "none"
        # "second from the top/bottom": the position list of the schema has no entry for it
        m = re.search(r"\bsecond\b.*\b(top|bottom)\b", text)
        if m and not re.search(r"\b(left|right)\b", text) and d["noun"] != "none":
            notes.append(f"set position second_from_{m.group(1)}")
            d["position"] = f"second_from_{m.group(1)}"
        if re.search(ALL_WORDS, text) and not t.get("all") and d["noun"] != "none":
            notes.append("set all (the instruction says all/every/both/a number)")
            t["all"] = True
        # "the gear that is not red": everything of that noun except the colour
        m = re.search(NOT_COLOUR, text)
        if m and d["noun"] in NOUNS:
            colour = "grey" if m.group(1) == "gray" else m.group(1)
            notes.append(f"not {colour}: excluded")
            t.setdefault("except", []).append({"words": colour, "colour": colour, "noun": d["noun"]})
            if d.get("colour") == colour:
                d["colour"] = "any"
        if t.get("to_noun", "none") != "none" and (not noun_said(t["to_noun"], text) or not re.search(PLACE_WORDS, text)):
            notes.append(f"dropped destination {t['to_noun']}")
            t["to_noun"] = "none"
        # "pick up the object lying on the grey part": "on the grey part" says which object, not where it goes
        if parsed.get("motion") == "pick_up" and t.get("to_noun", "none") != "none" and not re.search(PUT_VERBS, text):
            notes.append(f"dropped destination {t['to_noun']}: only picked up")
            t["to_noun"] = "none"
        for c, n in (("colour", "noun"), ("ref_colour", "ref_noun")):
            if not colour_said(d.get(c, "any"), d.get(n, "none"), text):
                notes.append(f"dropped {c} {d[c]}: not in front of {d[n]}")
                d[c] = "any"
        if not colour_said(t.get("to_colour", "any"), t.get("to_noun", "none"), text):
            notes.append(f"dropped to_colour {t['to_colour']}: not in front of {t['to_noun']}")
            t["to_colour"] = "any"
    # "put the two red gears in the tray": Apertus writes one task per gear; the same task twice is one task
    unique = []
    for t in tasks:
        if t not in unique:
            unique.append(t)
    if len(unique) < len(tasks):
        notes.append("merged identical tasks")
        tasks[:] = unique
    # "put the mandarin on the plate and leave it there": "it" is the object named before it
    named = next((t["target"] for t in tasks if t["target"]["noun"] != "none"), None)
    if named:
        tasks[:] = [t for t in tasks if t["target"]["noun"] != "none" or t.get("to_noun", "none") != "none"]
        for k in parsed.get("keep_still", []):
            if k["noun"] == "none" or (k.get("words") or "").strip().lower() in PRONOUNS:
                k["noun"], k["colour"] = named["noun"], named["colour"]
    parsed["tasks"] = tasks
    return parsed, notes


def missing(instruction: str, scene: Scene) -> str | None:
    """'pick up the blue cube' with no blue cube: refuse before asking Apertus. Every colour + noun
    pair of the instruction (and every noun) must fit some object, with the same tolerance as
    Scene.match, so a detection mistake in the shade does not turn into a refusal."""
    words = re.findall(r"[a-z]+", instruction.lower())
    for i, w in enumerate(words):
        noun = w[:-1] if w.endswith("s") and w[:-1] in NOUNS + list(GROUPS) else w
        if noun not in NOUNS and noun != "fruit":
            continue
        colour = words[i - 1] if i and words[i - 1] in COLOURS + ["gray"] else "any"
        colour = "grey" if colour == "gray" else colour
        if not scene.match(noun, colour):
            return f"There is no {noun if colour == 'any' else colour + ' ' + noun} on the table."
    return None


def unsafe_words(instruction: str) -> str | None:
    text = instruction.lower()
    return next((kind for pattern, kind in UNSAFE_WORDS if re.search(pattern, text)), None)


def refuse(iid: str, reason: str) -> dict:
    return {"instruction_id": iid, "target_pixel": None, "actions": [{"action": "refuse", "reason": reason}]}


def ask(iid: str, question: str) -> dict:
    return {"instruction_id": iid, "target_pixel": None, "actions": [{"action": "ask_user", "question": question}]}


def answer(iid: str, instruction: str, graph: dict, call) -> tuple[dict, dict]:
    """One instruction -> (result in the output schema, log of what happened)."""
    scene = Scene(graph)
    log = {"checks": []}
    reason = missing(instruction, scene)
    if reason:
        log["checks"].append(f"refuse: {reason}")
        return refuse(iid, reason), log
    log["parser"] = call(PARSER, f"Instruction: \"{instruction}\"", parser_schema())
    try:
        parsed, notes = repair(plain(json.loads(log["parser"])), instruction)
        log["checks"] += notes
    except json.JSONDecodeError:
        log["checks"].append("unparsable reply")
        return ask(iid, "I did not understand the instruction. Could you say it differently?"), log
    tasks = parsed.get("tasks") or []

    try:
        # 1. no object named at all ("give me that one"): ask before anything else
        if any(t["target"]["noun"] == "none" for t in tasks) or not tasks:
            raise NotDone("ask_user", "Which object do you mean? Please name it, for example \"the red cube\".")
        # 2. safety: the motion, according to Apertus and according to the word list
        kind = parsed.get("motion") if MOTIONS.get(parsed.get("motion")) else None
        word_kind = unsafe_words(instruction)
        if word_kind and word_kind != kind:
            log["checks"].append(f"word list: {word_kind}")
        # the word list names the reason when it fires: Apertus called "take the mug out of my hand" a throw
        if kind or word_kind:
            raise NotDone("refuse", MOTIONS[word_kind or kind])
        # 3. which objects, where to
        jobs, chosen = [], []
        for t in tasks:
            exclude = [o for e in t.get("except", []) for o in scene.match(e["noun"], e["colour"], scene.pickable)]
            objs = scene.resolve(t["target"], t.get("all", False), exclude + chosen)
            jobs.append((objs, scene.destination(t)))
            chosen += objs
        actions = build_plan(scene, jobs)
        # 4. objects that must stay where they are
        moved = [a["object"] for a in actions if a["action"] == "pick"]
        for k in parsed.get("keep_still", []):
            for o in scene.match(k["noun"], k["colour"], scene.pickable):
                if o["id"] not in moved:
                    continue
                if o in chosen:
                    raise NotDone("refuse", f"The instruction asks to move {name_of(o)} and also to leave it "
                                            "where it is; both cannot be done.")
                below = next((b for b in chosen if o in scene.tops(b)), chosen[0])
                raise NotDone("refuse", f"{name_of(o).capitalize()} lies on {name_of(below)}, so {name_of(below)} "
                                        f"cannot be picked without moving {name_of(o)}.")
    except NotDone as e:
        log["checks"].append(f"{e.kind}: {e.text}")
        return (refuse if e.kind == "refuse" else ask)(iid, e.text), log
    except (KeyError, TypeError) as e:   # a reply that is valid JSON but misses a field
        log["checks"].append(f"bad reply: {e!r}")
        return ask(iid, "I did not understand the instruction. Could you say it differently?"), log

    for a in actions:
        if a["object"] in scene.by_id:
            a["pixel"] = scene.by_id[a["object"]]["pixel"]
    return {"instruction_id": iid, "target_pixel": scene.by_id[actions[0]["object"]]["pixel"], "actions": actions}, log


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--graphs", type=Path, default=Path("outputs/scene_graphs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/results.json"))
    ap.add_argument("--base-url", default=BASE_URL, help="Ollama server")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--scene", help="only this scene, e.g. scene_06")
    ap.add_argument("--team", default="see-it-say-it-pick-it")
    ap.add_argument("--cache", type=Path, default=Path("outputs/cache/apertus.json"),
                    help="replies of Apertus by (model, prompt, instruction), so a change in the code after it "
                         "re-runs in seconds; temperature is 0, so a cached reply is the reply Apertus would give")
    ap.add_argument("--no-cache", action="store_true", help="ask Apertus again for every instruction")
    args = ap.parse_args()
    cache = {} if args.no_cache or not args.cache.exists() else json.loads(args.cache.read_text())

    def call(system: str, user: str, schema: dict) -> str:
        key = hashlib.sha256(json.dumps([args.model, system, user, schema]).encode()).hexdigest()
        if key not in cache:
            try:
                cache[key] = ask_apertus(args.base_url, args.model, system, user, schema, args.timeout)
            except (urllib.error.URLError, TimeoutError) as e:
                sys.exit(f"cannot reach Ollama at {args.base_url} ({e}). Is it running? Is {args.model} pulled?")
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(cache))
        return cache[key]

    scenes = sorted(p for p in args.data.iterdir() if p.is_dir() and (p / "rgb.png").exists())
    if args.scene:
        scenes = [p for p in scenes if p.name == args.scene]
    results, raw_path, t0 = [], args.out.with_suffix(".raw.jsonl"), time.time()
    with raw_path.open("w") as raw:
        for d in scenes:
            graph = json.loads((args.graphs / f"{d.name}.json").read_text())
            for iid, _, instruction in load_scene(d).instructions:
                result, log = answer(iid, instruction, graph, call)
                results.append(result)
                raw.write(json.dumps({"instruction_id": iid, "instruction": instruction, **log, "result": result}) + "\n")
                first = result["actions"][0]
                shown = " -> ".join(f"{a['action']} {a['object']}" for a in result["actions"]) if first["action"] in ("pick", "place") \
                    else f"{first['action']}: {first.get('question') or first.get('reason')}"
                print(f"{iid} \"{instruction}\"\n    {shown}", flush=True)
    args.out.write_text(json.dumps({"team": args.team, "results": results}, indent=2))
    print(f"\n{len(results)} answers in {args.out} in {time.time() - t0:.0f}s; raw replies in {raw_path}")


if __name__ == "__main__":
    main()
