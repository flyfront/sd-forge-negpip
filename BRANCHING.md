# Branch management

This repository maintains the following branches:

- `classic`: the upstream-compatible implementation for classic Forge.
- `neo-krea2-emphasis`: the Forge Neo implementation with Krea 2 support and the opt-in V-Scaling feature. This is the default branch and receives every Forge Neo change.
- `neo-moss`: `neo-krea2-emphasis` plus NegPiP for MOSS-Image, as provided by the third-party [moss-image-forge-neo](https://github.com/eduardoabreu81/moss-image-forge-neo) extension. It is kept apart because Forge Neo itself may gain MOSS-Image support with a different code path; it is not merged into `neo-krea2-emphasis`.

`neo-krea2` is frozen. It was the base shared by the Forge Neo variants until 2026-09-06; `neo-krea2-emphasis` behaves identically while V-Scaling is off, so the separate branch is no longer needed. Do not commit to it or merge it anywhere.

## Rules

1. Create a topic branch from the branch that will receive the change; do not commit directly to a maintained branch.
2. Start every Forge Neo change, including documentation, maintenance, fixes, and compatibility changes, from `neo-krea2-emphasis`, then merge it through a pull request.
3. Keep classic Forge changes on `classic`. Port an applicable change explicitly instead of merging a Neo branch into `classic`.
4. Leave `neo-krea2` untouched.
5. Keep MOSS-Image changes on `neo-moss` (`lib_negpip/moss.py` and the model detection in `scripts/negpip.py`); do not merge `neo-moss` into `neo-krea2-emphasis`. Bring `neo-krea2-emphasis` into `neo-moss` with a merge commit whenever it changes, so that `neo-moss` differs only by the MOSS-Image files.

## Typical flow

```console
git switch neo-krea2-emphasis
git switch -c fix/descriptive-name
# edit, verify, commit, and open a pull request targeting neo-krea2-emphasis
```

After merging into `neo-krea2-emphasis`:

```console
git switch neo-moss
git merge neo-krea2-emphasis
```
