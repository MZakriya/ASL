import requests
import os

url = "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"
filename = "rgb_imagenet.pt"

print(f"Downloading {filename} from {url}...")
try:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {filename} successfully ({os.path.getsize(filename)} bytes).")
except Exception as e:
    print(f"Failed to download: {e}")
