from PIL import Image
import os

def crop(image_path, coords):
    image_obj = Image.open(image_path)
    width, height = image_obj.size
    if width > 200:
	    cropped_image = image_obj.crop(coords)
	    cropped_image.save(image_path)


dataroot = '/home/wenwens/Documents/HumanPose/Pose-Transfer/fashion_data'
coords = (40, 0, 216, 256)

splits = ['train', 'test']

for split in splits:
	print(split)
	path = os.path.join(dataroot, split)
	images = os.listdir(path)
	for im in images:
		if im.endswith('.jpg'):
			crop(os.path.join(path, im), coords)




