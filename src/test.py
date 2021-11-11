import re
import os
import json

src_path = os.path.dirname(__file__)
json_path = os.path.join(src_path, 'settings', 'colorItems.json')

colorItems = {
    'left': {
        'Image': None,
        'Overlay image': (0, 255, 255, 255), #cyan
        'Text on segmented objects': (255, 255, 255, 255),
        'Contours of segmented objects': (255, 0, 0, 255),
        'Skeleton': (255, 0, 0, 255),
        'Contour': (255, 0, 0, 255)
    },
    'right': {
        'Image': None,
        'Overlay image': (255, 0, 255, 255), #magenta
        'Text on segmented objects': (255, 255, 255, 255),
        'Contours of segmented objects': (255, 0, 0, 255),
        'Skeleton': (255, 0, 0, 255),
        'Contour': (255, 0, 0, 255)
    }
}

print(colorItems)

# with open(json_path, mode='w') as file:
#     json.dump(colorItems, file, indent=2)

with open(json_path) as file:
    colorItems = json.load(file)

print(colorItems)
