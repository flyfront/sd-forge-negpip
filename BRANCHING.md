# Branch management

This repository maintains the following branches:

- `classic`: the upstream-compatible implementation for classic Forge (Haoming02's sd-forge-negpip, archived on 2026-09-30).
- `neo`: the Forge Neo implementation. This is the default branch and receives every Forge Neo change: SD1 / SDXL, Anima, Krea 2 with the opt-in V-Scaling, and MOSS-Image (Forge Neo's own support and the moss-image-forge-neo extension).
- `neo-krea2-emphasis`: a mirror of `neo`, kept so that installations made while it was the default branch keep updating. Push every `neo` commit to it; never commit to it directly.

Frozen branches, kept only so that existing checkouts do not break: `neo-krea2` (the base shared by the Forge Neo variants until 2026-09-06) and `neo-moss` (where the MOSS-Image support was developed until it was merged on 2026-10-22). Do not commit to them or merge them anywhere.

## Rules

1. Create a topic branch from `neo`; do not commit directly to a maintained branch.
2. Start every Forge Neo change, including documentation, maintenance, fixes, and compatibility changes, from `neo`, then merge it through a pull request.
3. After merging into `neo`, update the mirror: `git push origin neo:neo-krea2-emphasis`.
4. Keep classic Forge changes on `classic`. Port an applicable change explicitly instead of merging a Neo branch into `classic`.
5. Leave the frozen branches untouched.

## Typical flow

```console
git switch neo
git switch -c fix/descriptive-name
# edit, verify, commit, and open a pull request targeting neo
# after the merge:
git switch neo && git pull --ff-only
git push origin neo:neo-krea2-emphasis
```
