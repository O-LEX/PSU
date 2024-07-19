import numpy as np
import cv2

def pyramid(I, nlev=5, scale=0.5, sigma=1.0):
    """Generate Gaussian and Laplacian pyramids from an image."""
    G_pyr = [I]
    L_pyr = []
    
    filter_size = 5
    filter = cv2.getGaussianKernel(filter_size, sigma)
    filter = filter * filter.T
    
    for i in range(1, nlev):
        I_down = downsample(G_pyr[-1], filter, scale)
        I_up = upsample(I_down, filter, G_pyr[-1])
        L_pyr.append(G_pyr[-1] - I_up)
        G_pyr.append(I_down)

    return G_pyr, L_pyr

def downsample(I, filter, s):
    """Downsample the image using a Gaussian filter and resize."""
    O = cv2.filter2D(I, -1, filter, borderType=cv2.BORDER_REFLECT)
    O = cv2.resize(O, (int(I.shape[1] * s), int(I.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
    return O

def upsample(I, filter, Ref):
    """Upsample the image to match the reference size and apply Gaussian filter."""
    O = cv2.resize(I, (Ref.shape[1], Ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    O = cv2.filter2D(O, -1, filter, borderType=cv2.BORDER_REFLECT)
    return O

def bilateralFilter(A, G, w, sigma_s, sigma_r):
    """Apply joint bilateral filter to the image."""
    d = 2 * w + 1
    B = cv2.ximgproc.jointBilateralFilter(G, A, d, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)
    return B

def PyramidTextureFilter(I, sigma_s=5, sigma_r=0.05, nlev=11, scale=0.8):
    """Apply texture filtering using pyramid decomposition."""
    G, L = pyramid(I, nlev, scale)
    R = G[-1]
    
    for i in range(len(G) - 2, -1, -1):
        adaptive_sigma_s = sigma_s * (scale ** (i))
        w1 = int(np.ceil(adaptive_sigma_s * 0.5 + 1))
        w2 = int(np.ceil(adaptive_sigma_s * 2.0 + 1))
        
        R_up = cv2.resize(R, (G[i].shape[1], G[i].shape[0]), interpolation=cv2.INTER_LINEAR)
        R_hat = bilateralFilter(R_up, G[i], w1, adaptive_sigma_s, sigma_r)
        
        R_lap = R_hat + L[i]
        R_out = bilateralFilter(R_lap, R_hat, w2, adaptive_sigma_s, sigma_r)
        
        R_refine = bilateralFilter(R_out, R_out, w2, adaptive_sigma_s, sigma_r)
        R = R_refine
    
    return R

def main(image_path="asset/01.png", output_path="output.png"):
    """Main function to execute pyramid texture filtering."""
    I = cv2.imread(image_path)
    if I is None:
        raise ValueError("Image not found or the path is incorrect")
    I = I.astype(np.float32) / 255.0
    sigma_s = 20
    sigma_r = 0.09
    output = PyramidTextureFilter(I, sigma_s, sigma_r)
    output = (output * 255).astype(np.uint8)
    cv2.imwrite(output_path, output)

if __name__ == "__main__":
    main()
