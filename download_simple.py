import urllib.request
import ssl
import os

url = "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"
filename = "rgb_imagenet.pt"

print(f"Downloading {filename}...")
try:
    # Bypass SSL verification if needed (for some corp environments)
    context = ssl._create_unverified_context()
    urllib.request.urlretrieve(url, filename, context=context)
    print(f"Downloaded {filename} ({os.path.getsize(filename)} bytes).")
except Exception as e:
    print(f"Error: {e}")
