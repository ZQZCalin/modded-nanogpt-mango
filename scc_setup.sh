#!/bin/bash -l

module load python3/3.10.12 cuda/12.5

[ ! -d "env" ] && python -m venv env
source env/bin/activate

pip install -r requirements.txt
# SCC only installed cuda/12.2 and no cuda/12.6, 
# so we installed a different version of torch v2.7 that support cuda/12.2.
# pip install --pre torch==2.7.0.dev20250110+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
# pip install --pre torch==2.7.0.dev20250110+cu118 --index-url https://download.pytorch.org/whl/nightly --upgrade
# Now changed to cuda/12.5
pip install --pre torch==2.7.0.dev20250227+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade