# gemma3-tpu-pallas

## Python interpreter

Bare `python` / `python3` on this machine resolve to the Windows Store app-execution
alias, which exits with code 49 and the message "Python was not found". Do not call
them.

Always use the project's conda environment, addressed relative to the home directory:

```
"$HOME/.conda/envs/gemma3-tpu-pallas/python.exe"
```

That exact string works unchanged in both shells — Git Bash expands `$HOME` to
`/c/Users/<you>` and PowerShell expands it to `C:\Users\<you>`, and both resolve to the
same interpreter. In PowerShell it needs the call operator: `& "$HOME/.conda/envs/gemma3-tpu-pallas/python.exe" script.py`.
Keep the double quotes so the variable still expands while the path survives as one argument.

Do not create a venv or use `uv run` for project code: the environment already has the
JAX/Pallas stack installed, and a fresh interpreter will not.
