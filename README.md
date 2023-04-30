# Analyse an image

## installation instructions



System setup:

```bash
sudo apt update
sudo apt install libgl1-mesa-glx
```


Then:

1. Create a virtual environment
    ```bash
    python3 -m venv .venv
    ```
2. Activate the virtual environment
    ```bash
    source .venv/bin/activate
    ```
3. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
4. Run the script
    ```bash
    python3 analse_image.py
    ```
5. Deactivate the virtual environment
    ```bash
    deactivate
    ```