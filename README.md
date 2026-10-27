The files pyproject.toml and uv.lock in this folder describe which Python version and packages are used. They are shared by everyone.
Each person builds their own copy of the environment from those files. The environments are stored separately (in envs/<username>/), so nobody overwrites anyone else's.

Setup:
Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

Install direnv to set venv in different folders when you cd into it:
curl -sfL https://direnv.net/install.sh | bin_path=$HOME/.local/bin bash
printf '\neval "$(direnv hook bash)"\n' >> ~/.bashrc

Personal env folder:
mkdir -p $HOME/pl1358-01/project_data/envs/$USER

Setup uv storage to project_data as one command:
cat >> ~/.bashrc <<'EOF'

export UV_CACHE_DIR=$HOME/pl1358-01/project_data/envs/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=$HOME/pl1358-01/project_data/envs/$USER/python
EOF
source ~/.bashrc

Then:
cd into code/
and run to enable the per folder .envrc:
direnv allow

uv sync --locked
to update the environment to the same state
run 'uv add numpy' to add new packages (then commit)
