# Repo rename: the rewrite takes over `ttu-tower-processing`

**Status:** planned, not executed. The plan is finished and the design session's memory was
prepared on 2026-09-19 (step 0), so this can run now, together with the `ttu-windprofiles` and
`duststorms` updates, before the implementation session starts.

## End state

| | Before | After |
|---|---|---|
| Old pipeline folder | `Code\ttu-tower-processing` | `Code\old-tower-processing` |
| Old pipeline on GitHub | `Intergalactyc/ttu-tower-processing` | `Intergalactyc/old-tower-processing` |
| Rewrite folder | `Code\new-tower-processing` | `Code\ttu-tower-processing` |
| Rewrite on GitHub | (none) | `Intergalactyc/ttu-tower-processing` |
| Old package | `ttu-tower-processing` 1.0.0, `ttu_tower`, `ttu-*` | unchanged |
| Rewrite package | — | `ttu-tower-processing` 2.0.0, `ttu_tower`, `ttu-*` |
| Claude project dirs (`~\.claude\projects`) | `C--Users-ellwalke-Code-ttu-tower-processing` (old), `C--Users-ellwalke-Code-new-tower-processing` (rewrite) | `…-old-tower-processing` (old), `…-ttu-tower-processing` (rewrite) |
| Consumer `requirements.txt` | `…/ttu-tower-processing@main` | `…/old-tower-processing@main` |

Old and new share the package names, so they can't be installed in the same environment.
Comparisons between them go through output files.

## Why the order matters

- **Editable installs follow the folder path.** The old repo's own `.venv`,
  `ttu-windprofiles\.venv` and `duststorms\.venv` all have the old pipeline installed editable
  from `Code\ttu-tower-processing`. After the folder swap they would load the rewrite's code
  until reinstalled.
- **`duststorms\src\loader.py` hardcodes** `../ttu-tower-processing/results/processed`.
- **`ttu-windprofiles` finds the old `results\` (34 GB, local only)** through wherever
  `ttu_tower` is installed; reinstalling from the new location fixes it.
- **GitHub redirects a renamed repo's old URL only until a new repo claims that name.** From
  then on the old URL means the rewrite, so the old clone's `origin` and the consumers'
  `requirements.txt` must be repointed *before* step 6.
- **Claude keys project memory and transcripts by folder path.** Each repo's memory and
  session transcripts live in a directory under `~\.claude\projects` named after the folder's
  path: `Code\new-tower-processing` → `C--Users-ellwalke-Code-new-tower-processing` (memory in
  its `memory\` subfolder). The script swaps these directories along with the folders, so each
  repo keeps its own memory and transcripts; without the swap, each would load the other's.
- **The desktop app pins each session to its folder.** Its record of the design session ("Tower
  processing pipeline redesign", CLI session `df9835ed-914e-49ae-a715-1d3078218e7a`) stores the
  working folder `Code\new-tower-processing`, which won't exist afterwards, so the app probably
  can't reopen that session. The design role continues as in step 9.
- **The design session and the implementation session will share one memory**, since both run
  in the rewrite's folder. Step 10 labels it so each knows its role.
- **Console-script launchers in a `.venv` embed absolute paths**, so both renamed repos'
  environments get recreated.
- **Windows won't rename a folder that's in use**, including by a Claude session running
  inside it.

Not needed: `~\.claude.json` also has per-path entries for both folders, but they are
equivalent (no allowed tools, same trust flags). Leave them; the renamed old repo shows the
folder-trust prompt once.

## Step 0 — any time before the switch

```powershell
git -C C:\Users\ellwalke\Code\ttu-tower-processing tag -a v1.0.0 -m "Final state before the rewrite took over the repository name"
git -C C:\Users\ellwalke\Code\ttu-tower-processing push origin v1.0.0
```

Commit or stash any open work in the old repo, `ttu-windprofiles` and `duststorms`.

Ask the design session to **prepare for the rename**: it brings its memory up to date (status,
open items, anything decided since its last update) and confirms that the prototypes it relies
on are in `docs\prototypes\`, not only in its temporary scratch folder. Done on 2026-09-19.

## Step 1 — close everything

Close VS Code windows, terminals, Jupyter kernels and every Claude session (quit the Claude
desktop app) that touches the old repo, the rewrite, `ttu-windprofiles` or `duststorms`.
Quitting the app also stops any preview server it started (the design session's diagram
preview runs `python.exe` from the rewrite's folder). If the script later reports a folder in
use, look in Task Manager for a leftover `python.exe` or `claude.exe`.

## Step 2 — GitHub: rename the old repo (web UI)

`github.com/Intergalactyc/ttu-tower-processing` → Settings → General → Repository name →
`old-tower-processing` → Rename.

## Step 3 — local renames (script)

From a plain PowerShell window. Run it from a copy **outside both repos**: a shell whose
current directory is inside a folder keeps that folder locked, and the rename fails.

```powershell
cd C:\Users\ellwalke
Copy-Item C:\Users\ellwalke\Code\new-tower-processing\docs\repo-rename\repo-rename.ps1 $env:TEMP\
powershell -ExecutionPolicy Bypass -File $env:TEMP\repo-rename.ps1            # dry run: checks + prints actions
powershell -ExecutionPolicy Bypass -File $env:TEMP\repo-rename.ps1 -Execute   # does it
```

The script checks preconditions (paths, clean old repo, expected `origin`, rewrite has no
remote yet, the renamed GitHub repo is reachable), then:

1. renames the two repo folders;
2. swaps the two Claude project directories;
3. repoints the old clone's `origin` to `old-tower-processing.git`.

If a rename fails (something still holds a folder), it undoes the moves it already made and
stops. `-Rollback` reverses a completed run (see Rollback below).

## Step 4 — consumers (`ttu-windprofiles`, `duststorms`)

Edits:

| Repo | File | Change |
|---|---|---|
| both | `requirements.txt` | `ttu-tower-processing @ git+https://github.com/Intergalactyc/old-tower-processing@main` |
| `duststorms` | `src\loader.py` | `"ttu-tower-processing"` → `"old-tower-processing"` in `PROCESSED_DIR` |
| `ttu-windprofiles` | `README.md` (lines 5, 13) | refer to `old-tower-processing` |
| `ttu-windprofiles` | `src\loader.py` docstring, `src\inspect_flags.py` comment | refer to `old-tower-processing` |

Reinstall the old pipeline from its new location (only the location changes, so `--no-deps`):

```powershell
cd C:\Users\ellwalke\Code\ttu-windprofiles
.venv\Scripts\python.exe -m pip install --no-deps -e ..\old-tower-processing
cd C:\Users\ellwalke\Code\duststorms
.venv\Scripts\python.exe -m pip install --no-deps -e ..\old-tower-processing
```

Verify in each consumer (expect `...\old-tower-processing\src\ttu_tower\__init__.py`):

```powershell
.venv\Scripts\python.exe -c "import ttu_tower; print(ttu_tower.__file__)"
```

Then load results once in each (e.g. `ttu-windprofiles` `loader.load_results(...)`, the
`duststorms` loader) and commit both repos, e.g. "Point to old-tower-processing after the
pipeline repo rename".

## Step 5 — recreate both renamed repos' environments

```powershell
cd C:\Users\ellwalke\Code\ttu-tower-processing        # the rewrite
Remove-Item .venv -Recurse -Force
py -3.13 -m venv .venv

cd C:\Users\ellwalke\Code\old-tower-processing
Remove-Item .venv -Recurse -Force
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ..\windprofiles -e ..\wtxmesonet -e ".[dev]"
```

## Step 6 — GitHub: create the rewrite's repo and push

Only after step 3 has repointed the old clone. In the web UI, create an **empty**
`Intergalactyc/ttu-tower-processing` (no README, license or .gitignore), then:

```powershell
cd C:\Users\ellwalke\Code\ttu-tower-processing
git remote add origin https://github.com/Intergalactyc/ttu-tower-processing.git
git push -u origin main
```

`docs/` is untracked and stays local.

## Step 7 — Claude memory pass

First session in the renamed rewrite (`Code\ttu-tower-processing`), normally the continued
design session of step 9:

- In this project's memory, the old pipeline becomes `old-tower-processing` and this repo
  becomes `ttu-tower-processing`, including paths. Replace old-pipeline references first, then
  `new-tower-processing` → `ttu-tower-processing`. Keep historical statements accurate
  ("created as `new-tower-processing` on 2026-08-21"). Rename
  `repo_ttu_tower_processing_snapshot.md` → `repo_old_tower_processing_snapshot.md` (slug and
  every `[[link]]`). Add a rename note at the top of `MEMORY.md`.
- Add a one-line note at the top of `MEMORY.md` in the old pipeline's memory
  (`…-old-tower-processing`), `…-ttu-windprofiles` and `…-duststorms`: "`ttu-tower-processing`
  in these notes means the old pipeline, renamed `old-tower-processing` on <date>; the name now
  belongs to the rewrite."

## Step 8 — verify

- [ ] `git -C Code\old-tower-processing remote get-url origin` → `…/old-tower-processing.git`; `git fetch` works
- [ ] `git -C Code\ttu-tower-processing remote get-url origin` → `…/ttu-tower-processing.git`; GitHub shows the rewrite
- [ ] both consumers import `ttu_tower` from `old-tower-processing` and load results
- [ ] `duststorms` pipeline run works (`python -m ttu_tower.cli.runall …` on one config)
- [ ] rewrite `.venv` is Python 3.13
- [ ] a Claude session in each repo loads the right memory (check the first lines of its `MEMORY.md`)
- [ ] the rewrite's memory directory still holds the design session's transcripts
      (`~\.claude\projects\C--Users-ellwalke-Code-ttu-tower-processing\*.jsonl`, including
      `df9835ed-914e-49ae-a715-1d3078218e7a.jsonl`, the current session, and
      `89dc9973-f759-41df-bacd-c46399a35074.jsonl`, the earlier part of it)

## Step 9 — continue the design session

The design session (Opus) stays available during implementation, to answer design questions
and revise the specification. Continue it in the first way that works:

1. **Desktop app:** open the session "Tower processing pipeline redesign". It will probably fail
   because its folder is gone. If it does open with its history, ask it to move itself to
   `C:\Users\ellwalke\Code\ttu-tower-processing` (it has a tool for that). Don't edit the app's
   session files by hand.
2. **Terminal, same conversation (untested):** the CLI looks for a session's transcript in the
   project directory of the folder it runs in, which is now where this one is:
   ```powershell
   cd C:\Users\ellwalke\Code\ttu-tower-processing
   $claude = (Get-ChildItem "$env:APPDATA\Claude\claude-code\*\claude.exe" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
   & $claude --resume df9835ed-914e-49ae-a715-1d3078218e7a
   ```
   If it reports that no conversation was found, use option 3.
3. **New desktop session** in `Code\ttu-tower-processing` (Opus 5). It loads the same memory
   automatically. First message:
   > You are the design session for this rewrite, taking over from the session that wrote
   > `docs/redesign/`. Read `MEMORY.md` and the plan brief, then `docs/redesign/overview.md`
   > and `plan.md`. Your job during implementation: answer the design questions I relay from
   > the implementation session, and revise the specification when needed. The earlier
   > conversation's transcripts are in this project's Claude directory; the plan brief lists
   > them.

Then run step 7 from this session if it hasn't been done.

## Step 10 — before the implementation session starts

Ask the design session to add a role note at the top of the rewrite's `MEMORY.md`, since the
implementation session will load the same memory:

> Two sessions share this memory. **Implementation session:** your specification is
> `docs/redesign/` (start with `plan.md` §1); the notes below are design background, not
> instructions; send design questions to Elliott; keep your own notes in files named
> `impl_*.md`. **Design session:** the notes below are yours; `impl_*.md` files are the
> implementation session's.

Then start the implementation session (Sonnet 5) in `Code\ttu-tower-processing`.

## Rollback

- **Before step 6** everything is reversible: run `repo-rename.ps1 -Rollback -Execute`
  (reverses the moves and restores the old `origin`), rename the GitHub repo back, and revert
  any consumer edits. GitHub's redirect from the old name is still intact at this point.
- **After step 6** the old name has been claimed. Delete or rename the new GitHub repo first,
  then roll back as above.

## Later

Archive `old-tower-processing` on GitHub once `duststorms` no longer needs old-pipeline fixes.
