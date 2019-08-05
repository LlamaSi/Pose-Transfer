import os
from PIL import Image
from tqdm import tqdm

image_path = '../market_data/test_ori'
images = os.listdir(image_path)

target_path = '../market_data/test'
for filename in tqdm(images):
	if filename.endswith('.jpg'):
		img = Image.open(os.path.join(image_path, filename))
		img = img.resize((128,256), Image.ANTIALIAS)
		out = Image.new("RGB", (176, 256))
		out.paste(img, (0,0))

		# out.show()
		out.save(os.path.join(target_path, filename))
