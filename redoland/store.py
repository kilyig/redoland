"""Git-backed world store (Concordia engine).

Identical git mechanics to the standalone store: a run is its own git repo under
runs/<name>/, each year a commit tagged `<branch>-y<N>`, forking = a git branch
from a past year's tag. Only the *bodies* (core.Agent) + world globals are
serialized; the Concordia minds are rebuilt from bodies on load (lazily, via
world.mind()). The model_factory is injected at load time so a restored world can
think again — this is the substrate of fork/inject/replay.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Callable, Optional

from .core import Agent, Params, RNG
from .world import World

_SAFE = re.compile(r"[^a-z0-9_]+")


def safe_branch(name: str) -> str:
    return _SAFE.sub("_", name.lower()).strip("_") or "branch"


class GitStore:
    def __init__(self, path: str, model_factory: Optional[Callable[[int], object]] = None,
                 randomize_choices: bool = True):
        self.path = os.path.abspath(path)
        self.model_factory = model_factory
        self.randomize_choices = randomize_choices

    # -- git plumbing ----------------------------------------------------- #
    def _git(self, *args, check=True) -> str:
        r = subprocess.run(["git", *args], cwd=self.path, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr}")
        return r.stdout.strip()

    def init_repo(self):
        os.makedirs(self.path, exist_ok=True)
        if not os.path.isdir(os.path.join(self.path, ".git")):
            r = subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.path,
                               capture_output=True, text=True)
            if r.returncode != 0:
                self._git("init", "-q")
                self._git("checkout", "-q", "-b", "main")
            self._git("config", "user.email", "engine@redoland.local")
            self._git("config", "user.name", "Redoland Engine")
            self._git("config", "commit.gpgsign", "false")
            with open(os.path.join(self.path, ".gitignore"), "w") as fh:
                fh.write("live.jsonl\n")

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD")

    def tag(self, branch: str, year: int) -> str:
        return f"{safe_branch(branch)}-y{year}"

    def parse_tag(self, tag: str):
        m = re.match(r"^(.*)-y(\d+)$", tag)
        return (m.group(1), int(m.group(2))) if m else None

    def list_tags(self):
        parsed = []
        for t in self._git("tag").splitlines():
            p = self.parse_tag(t)
            if p:
                parsed.append((t, p[0], p[1]))
        return parsed

    def list_branches(self):
        out = self._git("for-each-ref", "--format=%(refname:short)", "refs/heads")
        return [b for b in out.splitlines() if b]

    def read_at(self, ref: str, relpath: str) -> Optional[str]:
        r = subprocess.run(["git", "show", f"{ref}:{relpath}"], cwd=self.path,
                           capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    def list_dir_at(self, ref: str, relpath: str):
        out = subprocess.run(["git", "ls-tree", "--name-only", f"{ref}:{relpath}"],
                             cwd=self.path, capture_output=True, text=True)
        return [l for l in out.stdout.splitlines() if l] if out.returncode == 0 else []

    # -- world <-> files -------------------------------------------------- #
    def write_world(self, world: World):
        meta = {
            "year": world.year, "branch": world.branch, "pile": world.pile,
            "next_eid": world.next_eid, "next_aid": world.next_aid,
            "used_names": sorted(world.used_names),
            "rng_state": world.rng.get_state(), "params": world.params.to_dict(),
        }
        self._write_json("meta.json", meta)
        adir = os.path.join(self.path, "agents")
        os.makedirs(adir, exist_ok=True)
        for f in os.listdir(adir):
            if f.endswith(".json"):
                os.remove(os.path.join(adir, f))
        for aid, ag in world.agents.items():
            self._write_json(f"agents/{aid}.json", ag.to_dict())
        if world.year_events:
            edir = os.path.join(self.path, "events")
            os.makedirs(edir, exist_ok=True)
            with open(os.path.join(edir, f"year-{world.year}.jsonl"), "w") as fh:
                for ev in world.year_events:
                    fh.write(json.dumps(ev) + "\n")

    def load_world(self, ref: Optional[str] = None) -> World:
        if ref is None:
            meta = json.loads(open(os.path.join(self.path, "meta.json")).read())
            names = [f for f in os.listdir(os.path.join(self.path, "agents")) if f.endswith(".json")]
            agents = [json.loads(open(os.path.join(self.path, "agents", f)).read()) for f in names]
        else:
            meta = json.loads(self.read_at(ref, "meta.json"))
            names = self.list_dir_at(ref, "agents")
            agents = [json.loads(self.read_at(ref, f"agents/{f}")) for f in names]
        params = Params.from_dict(meta["params"])
        rng = RNG(state=meta["rng_state"])
        world = World(params, rng, branch=meta["branch"],
                      model_factory=self.model_factory,
                      randomize_choices=self.randomize_choices)
        world.year = meta["year"]
        world.pile = meta["pile"]
        world.next_eid = meta["next_eid"]
        world.next_aid = meta["next_aid"]
        world.used_names = set(meta["used_names"])
        for ad in agents:
            world.agents[ad["id"]] = Agent.from_dict(ad)
        return world                                  # minds rebuilt lazily via world.mind()

    def _write_json(self, relpath, obj):
        full = os.path.join(self.path, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)

    # -- commits / tags --------------------------------------------------- #
    def commit_year(self, world: World, message: str, retag: bool = False,
                    tag_year: bool = True):
        self.write_world(world)
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", message)
        if not tag_year:           # genesis / founding state — committed but not a "year"
            return
        tag = self.tag(world.branch, world.year)
        if retag:
            self._git("tag", "-f", tag)
        else:
            subprocess.run(["git", "tag", tag], cwd=self.path, capture_output=True, text=True)

    def checkout_fork(self, parent_branch: str, year: int, new_branch: str):
        src = self.tag(parent_branch, year)
        nb = safe_branch(new_branch)
        self._git("checkout", "-q", "-b", nb, src)
        subprocess.run(["git", "tag", self.tag(nb, year)], cwd=self.path,
                       capture_output=True, text=True)
        return nb

    def checkout_branch(self, branch: str):
        self._git("checkout", "-q", safe_branch(branch))

    def years_for_branch(self, branch: str):
        b = safe_branch(branch)
        return sorted(y for (_, bb, y) in self.list_tags() if bb == b)
