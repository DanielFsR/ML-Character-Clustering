# setup.sh
#!/usr/bin/env bash

# 1. Create & activate a virtual environment
ENV_NAME=venv
python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

# 2. Install all required libraries from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the clustering script on the default input
#    Usage: ./setup.sh [input_file]
if [ -z "$1" ]; then
  INPUT_FILE="input.txt"
else
  INPUT_FILE="$1"
fi

echo "Running clustering on $INPUT_FILE..."
python cluster_chars.py "$INPUT_FILE"
