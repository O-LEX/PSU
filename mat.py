import numpy as np
import cv2

def pyramid(I, nlev=5, scale=0.5, sigma=1.0):
    r, c = I.shape[:2]

    # Recursively build pyramid
    G_pyr = [None] * nlev
    L_pyr = [None] * (nlev - 1)
    filter = cv2.getGaussianKernel(5, sigma) * cv2.getGaussianKernel(5, sigma).T

    G_pyr[0] = I

    for l in range(1, nlev):
        I_down = downsample(I, filter, scale)
        L_pyr[l - 1] = I - upsample(I_down, filter, I.shape)
        G_pyr[l] = I_down
        I = I_down

    return G_pyr, L_pyr

def downsample(I, filter, scale):
    I_filtered = cv2.filter2D(I, -1, filter, borderType=cv2.BORDER_REFLECT)
    r, c = I.shape[:2]
    new_size = (int(c * scale), int(r * scale))
    O = cv2.resize(I_filtered, new_size, interpolation=cv2.INTER_LINEAR)
    return O

def upsample(I, filter, ref_shape):
    r, c = ref_shape[:2]
    O = cv2.resize(I, (c, r), interpolation=cv2.INTER_LINEAR)
    O_filtered = cv2.filter2D(O, -1, filter, borderType=cv2.BORDER_REFLECT)
    return O_filtered

def bilateral_filter(A, G, w, sigma_s, sigma_r):
    # Pre-compute the Gaussian spatial kernel
    Gs = cv2.getGaussianKernel(2 * w + 1, sigma_s) * cv2.getGaussianKernel(2 * w + 1, sigma_s).T

    # Parameters
    dimX, dimY, c = A.shape  # dimensions of A

    # Initialize the image
    B = np.zeros_like(A)

    def process_pixel(i, j):
        # Extract the local patch
        minX = max(i - w, 0)
        maxX = min(i + w, dimX - 1)
        minY = max(j - w, 0)
        maxY = min(j + w, dimY - 1)
        PA = A[minX:maxX+1, minY:maxY+1, :]
        PG = G[minX:maxX+1, minY:maxY+1, :]

        # Compute the Gaussian range kernel
        dR = PG[:, :, 0] - G[i, j, 0]
        dG = PG[:, :, 1] - G[i, j, 1]
        dB = PG[:, :, 2] - G[i, j, 2]
        Gr = np.exp(-(dR**2 + dG**2 + dB**2) / (2 * sigma_r**2))

        # Compute the bilateral filter response
        F = Gr * Gs[minX-i+w:maxX-i+w+1, minY-j+w:maxY-j+w+1]
        normF = np.sum(F)
        B[i, j, :] = np.sum(PA * F[:, :, np.newaxis], axis=(0, 1)) / normF

    return B

def pyramid_texture_filter(I, sigma_s=5, sigma_r=0.05, nlev=11, scale=0.8):
    G, L = pyramid(I, nlev, scale)

    R = G[-1]
    for l in range(len(G)-2, -1, -1):
        adaptive_sigma_s = sigma_s * (scale ** (l))
        w1 = int(np.ceil(adaptive_sigma_s * 0.5 + 1))
        w2 = int(np.ceil(adaptive_sigma_s * 2.0 + 1))

        # Upsample
        R_up = cv2.resize(R, (G[l].shape[1], G[l].shape[0]), interpolation=cv2.INTER_LINEAR)
        R_hat = bilateral_filter(R_up, G[l], w1, adaptive_sigma_s, sigma_r)

        # Laplacian
        R_lap = R_hat + L[l]
        R_out = bilateral_filter(R_lap, R_hat, w2, adaptive_sigma_s, sigma_r)

        # Enhancement
        R_refine = bilateral_filter(R_out, R_out, w2, adaptive_sigma_s, sigma_r)

        R = R_refine

    return R

# Example usage:
I = cv2.imread('asset/01.png')
R = pyramid_texture_filter(I)
cv2.imwrite('output.png', R)
