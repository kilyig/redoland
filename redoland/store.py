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
        # `.git` is a directory in a normal repo but a FILE in a worktree or submodule
        if not os.path.exists(os.path.join(self.path, ".git")):
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

    def adopt_clone(self):
        """Make a fresh clone usable as a run. A `git clone` / submodule checkout carries
        the other worldlines only as remote-tracking branches (origin/eron8, ...) — at most
        the default branch exists locally — and a submodule's HEAD is detached, so
        list_branches() (refs/heads) would miss them. Create a local branch for every
        origin/* branch that has none, and if HEAD is detached exactly at the tip of
        `main` (else the first such branch) attach it there. Idempotent and cheap: a repo
        with no `origin` — every normal run — returns after a single git call; a worktree
        shares refs/heads with its main repo, so nothing is created there either. A HEAD
        detached by _free_branch_from_main is left alone (its branch has moved on in a
        worktree, or is held by one, in which case the checkout simply fails)."""
        if not os.path.exists(os.path.join(self.path, ".git")):
            return
        out = self._git("for-each-ref", "--format=%(refname:strip=3)", "refs/remotes/origin/",
                        check=False)
        remote = [b for b in out.splitlines() if b and b != "HEAD"]
        if not remote:
            return
        local = set(self.list_branches())
        for b in remote:
            if b not in local:
                subprocess.run(["git", "branch", "-q", b, f"origin/{b}"], cwd=self.path,
                               capture_output=True, text=True)
        if self._git("symbolic-ref", "-q", "HEAD", check=False) == "":     # detached
            target = "main" if "main" in remote else remote[0]
            if self._git("rev-parse", "HEAD") == self._git("rev-parse", target, check=False):
                subprocess.run(["git", "checkout", "-q", target], cwd=self.path,
                               capture_output=True, text=True)

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
            "cursor": world.cursor,            # resumable run position (phase/last/steps)
            "premise": world.premise,          # the world's "stage" (creator's backstory)
            "next_pile_bonus": world.next_pile_bonus,   # food rolling into next year's pile
            "next_pile_set": world.next_pile_set,       # injected override of next harvest
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
                      randomize_choices=self.randomize_choices,
                      premise=meta.get("premise", ""))
        world.year = meta["year"]
        world.pile = meta["pile"]
        world.next_eid = meta["next_eid"]
        world.next_aid = meta["next_aid"]
        world.next_pile_bonus = meta.get("next_pile_bonus", 0)
        world.next_pile_set = meta.get("next_pile_set")        # absent in older runs -> None
        world.used_names = set(meta["used_names"])
        # resumable cursor — default to a clean year boundary (so old year-only
        # commits, which have no cursor, continue correctly into the next year).
        world.cursor = meta.get("cursor") or {"phase": "year_start", "last": None, "steps": 0}
        for ad in agents:
            world.agents[ad["id"]] = Agent.from_dict(ad)
        # reload this year's accumulated events so a mid-year resume — or an inject at
        # a boundary — keeps the cumulative transcript intact (each commit rewrites
        # events/year-<N>.jsonl; a fresh year resets it in step()'s year_start phase).
        raw = (self.read_at(ref, f"events/year-{world.year}.jsonl") if ref is not None
               else self._read_local(f"events/year-{world.year}.jsonl"))
        if raw:
            world.year_events = [json.loads(l) for l in raw.splitlines() if l.strip()]
        return world                                  # minds rebuilt lazily via world.mind()

    def _read_local(self, relpath: str) -> Optional[str]:
        p = os.path.join(self.path, relpath)
        return open(p).read() if os.path.exists(p) else None

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

    def checkout_fork_at(self, commit: str, new_branch: str):
        """Branch a new worldline from ANY commit hash (a single action), not just a
        year tag — this is what lets the AI fork from within any step."""
        nb = safe_branch(new_branch)
        self._git("checkout", "-q", "-b", nb, commit)
        return nb

    def checkout_branch(self, branch: str):
        self._git("checkout", "-q", safe_branch(branch))

    # -- worktrees (concurrency: one working tree per running branch) ------ #
    # A git repo has a single working tree, so two branches of the SAME run cannot be
    # advanced at once through it. Linked worktrees give each running branch its own
    # working dir while sharing the object DB and refs — so commits made in a worktree
    # are immediately visible to ref-based reads (git show <branch>:path) from the main
    # repo. This is what lets multiple branches of one world run concurrently.
    def _free_branch_from_main(self, b: str):
        """If `b` is the branch checked out in the MAIN working tree, detach it so a
        worktree can claim it (reads are all ref-based, so a detached main tree is
        fine). Uniformly running every branch in a worktree keeps the model simple."""
        cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.path,
                             capture_output=True, text=True).stdout.strip()
        if cur == b:
            subprocess.run(["git", "checkout", "--detach", "-q"], cwd=self.path,
                           capture_output=True, text=True)

    def ensure_worktree(self, branch: str, wt_path: str) -> str:
        """Create (or reuse) a linked worktree with `branch` checked out at `wt_path`,
        and return it. Idempotent: an existing worktree dir is reused as-is."""
        b = safe_branch(branch)
        if os.path.exists(os.path.join(wt_path, ".git")):   # .git is a FILE in a worktree
            return wt_path
        os.makedirs(os.path.dirname(wt_path), exist_ok=True)
        subprocess.run(["git", "worktree", "prune"], cwd=self.path,
                       capture_output=True, text=True)
        self._free_branch_from_main(b)
        r = subprocess.run(["git", "worktree", "add", "-f", wt_path, b],
                           cwd=self.path, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git worktree add ({b}) failed:\n{r.stderr.strip()}")
        return wt_path

    def current_commit(self) -> str:
        return self._git("rev-parse", "HEAD")

    def timeline(self, branch: Optional[str] = None, n: int = 40):
        """Recent action-commits as [{commit, desc}] — the per-step history the AI
        scans to choose a fork point (each desc is the step's main event text)."""
        args = ["log", f"-{int(n)}", "--format=%h\t%s"]
        if branch:
            args.append(safe_branch(branch))
        rows = []
        for ln in self._git(*args).splitlines():
            if "\t" in ln:
                h, s = ln.split("\t", 1)
                rows.append({"commit": h, "desc": s})
        return rows

    def years_for_branch(self, branch: str):
        b = safe_branch(branch)
        return sorted(y for (_, bb, y) in self.list_tags() if bb == b)
