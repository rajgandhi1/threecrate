# Skill evals

Ten cases for `claude plugin eval`. Nine give the agent a sample scan from `samples/`
and check the result against its known answers. One checks the skill stays out of
unrelated requests.

Run from `skills/` on Linux or WSL2 (Bash runs in a sandbox that native Windows lacks),
with `threecrate` 0.12.0 or newer installed:

```bash
claude plugin eval . --scaffold --allow-tools Bash Write
```

`--scaffold` lets each case copy its sample scan into the workspace. Results go to
`evals/results/`, which git ignores.
