# Project: Epidemic Network Experiments

This folder contains refactored code from a single large `Main.py` into a small
package named `epinet` so the code is easier to maintain and commit to Git.

Package layout and files:

- `epinet/` (package)
  - `__init__.py` — package exports and convenience imports
  - `network.py` — Network class and graph generation helpers
  - `epidemic.py` — Epidemic simulation and disjoint-set outbreak estimation
  - `assortativity.py` — Functions to build assortative/disassortative networks
  - `metrics.py` — Vaccination and probability computations
  - `plotting.py` — Plotting helpers and the infection_length_vs_clustering_ws routine
- `Main.py` — Small entrypoint that demonstrates a simple run using the package
- `requirements.txt` — Python dependencies

How to run (Windows PowerShell) from the project folder:

```powershell
python Main.py
```

Alternatively, to use `epinet` as a package in other projects, you can
install it in editable mode (recommended during development):

```powershell
# from the project root
python -m pip install -e .
```

If you don't install the package, running `Main.py` from the package root
works because Python will add the current directory to `sys.path`.

Notes:

- If your editor shows "unresolved import" warnings for the new package,
  open the project folder as the workspace root or configure the Python
  path in your editor to include the project root. Running `Main.py`
  directly will work from the folder.
