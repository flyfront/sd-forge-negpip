# Branch management

This repository maintains the following branches:

- `classic`: the upstream-compatible implementation for classic Forge.
- `neo-krea2`: the base implementation shared by the Forge Neo Krea 2 variants.
- `neo-krea2-emphasis`: `neo-krea2` plus the opt-in V-Scaling feature. This is the default branch.

## Rules

1. Create a topic branch from the branch that will receive the change; do not commit directly to a maintained branch.
2. Branch shared Krea 2 fixes and compatibility changes from `neo-krea2`, then merge them through a pull request.
3. Merge `neo-krea2` into `neo-krea2-emphasis` after the shared change is verified.
4. Branch V-Scaling-specific changes from `neo-krea2-emphasis` and merge them only into that branch.
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
```
