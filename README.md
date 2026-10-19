The files pyproject.toml and uv.lock in this folder describe which Python version and packages are used. They are shared by everyone.
Each person builds their own copy of the environment from those files. The environments are stored separately (in envs/<username>/), so nobody overwrites anyone else's.

Setup:
Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

Personal env folder:
mkdir $HOME/pl1358-01/project_data/envs/$USER

Setup uv storage to project_data:
echo 'export UV_PROJECT_ENVIRONMENT=$HOME/pl1358-01/project_data/envs/$USER/venv' >> ~/.bashrc
echo 'export UV_CACHE_DIR=$HOME/pl1358-01/project_data/envs/$USER/uv-cache' >> ~/.bashrc
echo 'export UV_PYTHON_INSTALL_DIR=$HOME/pl1358-01/project_data/envs/$USER/python' >> ~/.bashrc
source ~/.bashrc

Then:
uv sync --locked
to update the environment to the same state
run 'uv add numpy' to add new packages (then commit)
