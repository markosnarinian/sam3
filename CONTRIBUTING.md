# Contributing to sam3
We want to make contributing to this project as easy and transparent as
possible.

## Development environment

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then
create the locked development and training environment from the repository root:

```bash
uv sync --locked --group dev --extra train
```

Run tools through `uv` so they use the managed `.venv`:

```bash
uv run pytest -q test
uv run ufmt format <changed-python-files>
```

When changing dependencies, update `pyproject.toml`, run `uv lock`, and include
the resulting `uv.lock` change. Otherwise, use `uv lock --check` to confirm the
lockfile is current.

### Amp orbs

`.agents/setup` prepares the locked development and training environment, Python
3.12, and system libraries for video processing. Amp snapshots this environment
for reuse by fresh orbs. `.agents/resume` checks the existing environment without
installing anything on wake. To repair or refresh dependencies manually, rerun
`.agents/setup`.

Use `uv run --no-sync pytest -q test` and `uv run --no-sync ufmt format <files>`
to use the prepared environment without changing its installed feature set.
Notebook and evaluation dependencies are not preinstalled. GPU-only acceleration
extensions, NVIDIA drivers, CUDA compilation tools, model weights, and Hugging
Face authentication are also excluded. CPU tests run in orbs, but full inference
and training require suitable GPU hardware and authorized model access.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Make sure your code lints.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to sam3, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
