# Analyse an image

## installation instructions



System setup:

1. Assuming you are on ubuntu-linux/similar use these commands:

```bash
sudo apt update
sudo apt install libgl1-mesa-glx

echo -e "VISION_KEY=<YOUR_KEY>\nVISION_ENDPOINT=<YOUR_ENDPOINT>" > .env

```

2. Then fill in the details (API Key and endpoint) of your Azure Cognitive Services Vision instance in the .env file.


3. Install the python dependencies:

```bash
pip install -r requirements.txt
```

4. Run the code with the example files:

```bash
python3 analse_image.py
```
