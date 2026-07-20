# Project Conventions

## Python Research Scripts

- Write Python code in a readable research-script style: clear top-level configuration, straightforward data flow, and descriptive variable names.
- Favor direct, easy-to-read code over heavy engineering wrappers, deep abstractions, or unnecessary framework-style structure.
- Add simple comments in Python code when they help explain the scientific intent or a non-obvious processing step.
- Put user-adjustable settings near the top of each script in a clearly labeled configuration block.
- Do not use `argparse`, `ArgumentParser`, or `parser.add_argument` for project scripts unless the user explicitly asks for a command-line interface.
- Prefer changing behavior by editing constants in the script's configuration block, such as input paths, output paths, leads, sample counts, plotting options, and feature flags.
- Keep scripts runnable directly with `python script_name.py`; avoid hidden setup steps when a simple top-level configuration is enough.
- Keep helper functions small and named for the scientific operation they perform.
- Preserve existing project patterns for figures, paths, and result outputs unless a requested change needs a different structure.


## Python Environment

- Use the project's available PyTorch Python environment when running Python scripts or analysis for this project.
- Prefer commands that make the selected PyTorch environment explicit when there is any ambiguity about which Python interpreter is active.

