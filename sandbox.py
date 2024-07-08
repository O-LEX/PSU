import cv2
import numpy as np

image_path = "asset/01.png"
image = cv2.imread(image_path)
image = image.astype(np.float32) / 255.0
image = cv2.resize(image, (image.shape[1]//(2**5), image.shape[0]//(2**5)), interpolation=cv2.INTER_LINEAR)
image = (image * 255).astype(np.uint8)
cv2.imwrite("output.png", image)
