import os
import re
import sys

from PIL import Image

# Crop a batch of images based on largest common alpha box
# Usage: python cropall.py <images...>
# Output: crop/<images>.png

assert len(sys.argv) > 1, 'No images specified'
patterns = [ re.compile(p) for p in sys.argv[1:] ]

images = {}
for root, dirs, files in os.walk('.'):
    for f in files:
        for p in patterns:
            if p.match(f):
                path = os.path.join(root, f)
                images[path] = Image.open(path)

print('Cropping images', images)

# Get largest common alpha box
boxes = [ image.getbbox() for image in images.values() ]
print('Boxes', boxes)

left = min([ box[0] for box in boxes ])
top = min([ box[1] for box in boxes ])
right = max([ box[2] for box in boxes ])
bottom = max([ box[3] for box in boxes ])

print('Common box', left, top, right, bottom)

# Crop images
os.makedirs('crop', exist_ok=True)
for path, image in images.items():
    image = image.crop((left, top, right, bottom))
    image.save('crop/' + os.path.basename(path))
