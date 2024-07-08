import cv2
import numpy as np

def joint_bilateral_upsample(guide, src, d, sigma_color, sigma_space):
    upsampled = cv2.resize(src, (guide.shape[1], guide.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.ximgproc.jointBilateralFilter(guide, upsampled, d, sigma_color, sigma_space)

def PSU(image, n, sigma_s, sigma_r):
    image = image.astype(np.float32) / 255.0
    output = cv2.resize(image, (image.shape[1]//(2**n), image.shape[0]//(2**n)), interpolation=cv2.INTER_LINEAR)
    
    for i in range(n):
        guide = cv2.resize(image, (image.shape[1]//(2**(n-1-i)), image.shape[0]//(2**(n-1-i))), interpolation=cv2.INTER_LINEAR)
        temp = cv2.Laplacian(guide, cv2.CV_32F, ksize=3)
        guide = joint_bilateral_upsample(guide, output, d=9, sigma_color=75, sigma_space=75)
        temp += guide
        output = cv2.ximgproc.jointBilateralFilter(guide, temp, d=9, sigmaColor=75, sigmaSpace=75)

    output = (output * 255).astype(np.uint8)
    return output


def main():
    image_path = "asset/01.png"
    image = cv2.imread(image_path)
    n = 2
    sigma_s = 3
    sigma_r = 0.1
    output = PSU(image, n, sigma_s, sigma_r)
    cv2.imwrite("output.png", output)

if __name__ == "__main__":
    main()
