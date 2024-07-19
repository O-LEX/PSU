import cv2
import numpy as np

def joint_bilateral_upsample(guide, src, d, sigmaColor, sigmaSpace):
    upsampled = cv2.resize(src, (guide.shape[1], guide.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.ximgproc.jointBilateralFilter(guide, upsampled, d, sigmaColor, sigmaSpace)

def pyramid_guided_structure_aware_upsample(guide, src, laplacian, d, sigmaColor, sigmaSpace):
    # Joint bilateral upsample
    upsampled = joint_bilateral_upsample(guide, src, d, sigmaColor, sigmaSpace)
    # Add Laplacian details and apply joint bilateral filter again
    detail_enhanced = upsampled + laplacian
    return cv2.ximgproc.jointBilateralFilter(upsampled, detail_enhanced, d, sigmaColor, sigmaSpace)

def PSU(image, n, sigma_s, sigma_r):
    image = image.astype(np.float32) / 255.0
    
    # Build Gaussian and Laplacian pyramids
    gaussian_pyramid = [image]
    laplacian_pyramid = []
    
    for i in range(n):
        downsampled = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(downsampled)
        upsampled = cv2.pyrUp(downsampled, dstsize=(gaussian_pyramid[-2].shape[1], gaussian_pyramid[-2].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[-2], upsampled)
        laplacian_pyramid.append(laplacian)
    
    # Start with the coarsest Gaussian pyramid level
    output = gaussian_pyramid[-1]

    for i in range(n-1, -1, -1):
        guide = gaussian_pyramid[i]
        laplacian = laplacian_pyramid[i]
        output = pyramid_guided_structure_aware_upsample(guide, output, laplacian, d=9, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    
    output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)
    output = (output * 255).astype(np.uint8)
    return output

def main():
    image_path = "asset/02.png"
    image = cv2.imread(image_path)
    n = 5
    sigma_s = 5
    sigma_r = 0.07
    output = PSU(image, n, sigma_s, sigma_r)
    cv2.imwrite("output.png", output)

if __name__ == "__main__":
    main()
