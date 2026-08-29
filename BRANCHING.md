# Branch management

This repository maintains the following branches:

- `classic`: the upstream-compatible implementation for classic Forge.
- `neo-krea2`: the base implementation shared by the Forge Neo Krea 2 variants.
- `neo-krea2-emphasis`: `neo-krea2` plus the opt-in V-Scaling feature. This is the default branch.

## Rules

1. Create a topic branch from the branch that will receive the change; do not commit directly to a maintained branch.
2. Start every change intended for both Krea 2 branches from `neo-krea2`, including documentation, maintenance, fixes, and compatibility changes, then merge it through a pull request.
3. Do not implement or commit the shared part independently on both Krea 2 branches. After the shared change is verified and merged into `neo-krea2`, merge `neo-krea2` into `neo-krea2-emphasis`.
4. If the change also needs V-Scaling-specific adjustments, add them only after the shared change has reached `neo-krea2-emphasis`. Branch those adjustments from the updated `neo-krea2-emphasis` and merge them only into that branch.
5. Do not merge `neo-krea2-emphasis` back into `neo-krea2`.
6. Keep classic Forge changes on `classic`. Port an applicable change explicitly instead of merging a Neo branch into `classic`.

## Typical flow

```console
git switch neo-krea2
git switch -c fix/descriptive-name
# edit, verify, commit, and open a pull request targeting neo-krea2

# after merging the pull request
git switch neo-krea2-emphasis
git merge neo-krea2
# verify the combined branch

# if needed, add an emphasis-only follow-up after the merge
git switch -c change/descriptive-emphasis-update
# edit, verify, commit, and open a pull request targeting neo-krea2-emphasis
```
