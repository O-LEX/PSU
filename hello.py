import cv2

def joint_bilateral_upsample(guide, src, d, sigma_color, sigma_space):
    upsampled = cv2.resize(src, (guide.shape[1], guide.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.ximgproc.jointBilateralFilter(guide, upsampled, d, sigma_color, sigma_space)

def PSU(image_path, n):
    image = cv2.imread(image_path)
    output = cv2.resize(image, (image.shape[1]//(2**n), image.shape[0]//(2**n)), interpolation=cv2.INTER_LINEAR)
    
    for i in range(n):
        guide = cv2.resize(image, (image.shape[1]//(2**(n-1-i)), image.shape[0]//(2**(n-1-i))), interpolation=cv2.INTER_LINEAR)
        output = joint_bilateral_upsample(guide, output, d=9, sigma_color=75, sigma_space=75)
        guide = cv2.Laplacian(guide, cv2.CV_64F)

    
    return output

def main():
    image_path = "asset/01.png"
    n = 3
    output = PSU(image_path, n)
    cv2.imwrite("output.png ", output)

if __name__ == "__main__":
    main()


def 