import cv2
import numpy as np

# 画像のパス
image_path = "asset/02.png"

# 画像の読み込み
image = cv2.imread(image_path)

image = image.astype(np.float32) / 255.0

# ピラミッドのレベル数
n = 11

# 画像のコピーを作成
G = image.copy()
gpA = [G]

# Gaussianピラミッドの作成
for i in range(n):
    G = cv2.pyrDown(G)
    gpA.append(G)

# Laplacianピラミッドの作成
lpA = []
for i in range(n):
    GE = cv2.pyrUp(gpA[i+1])
    GE = cv2.resize(GE, (gpA[i].shape[1], gpA[i].shape[0]))  # サイズを合わせる
    L = cv2.subtract(gpA[i], GE)
    lpA.append(L)

# 画像の表示
# cv2.imshow('Original Image', image)
# for i in range(n):
#     cv2.imshow(f'Gaussian Level {i}', gpA[i])
#     cv2.imshow(f'Laplacian Level {i}', lpA[i])

# cv2.waitKey(0)
# cv2.destroyAllWindows()

sigmaColor = 0.07
sigmaSpace = 5
scale=0.8
src = gpA[-1]
for i in range(n):
    adaptive_sigmaSpace = sigmaSpace * (scale ** (n-i))
    w1 = int(np.ceil(adaptive_sigmaSpace * 0.5 + 1))
    w2 = int(np.ceil(adaptive_sigmaSpace * 2.0 + 1))
    src = cv2.resize(src, (gpA[n-i-1].shape[1], gpA[n-i-1].shape[0]), interpolation=cv2.INTER_LINEAR)
    guide = cv2.ximgproc.jointBilateralFilter(gpA[n-i-1], src, d=2*w1+1, sigmaColor=sigmaColor, sigmaSpace=adaptive_sigmaSpace, borderType=cv2.BORDER_REPLICATE)
    src = guide + lpA[n-i-1]
    src = cv2.ximgproc.jointBilateralFilter(guide, src, d=2*w2+1, sigmaColor=sigmaColor, sigmaSpace=adaptive_sigmaSpace, borderType=cv2.BORDER_REPLICATE)
    src = cv2.ximgproc.jointBilateralFilter(src, src, d=2*w2+1, sigmaColor=sigmaColor, sigmaSpace=adaptive_sigmaSpace, borderType=cv2.BORDER_REPLICATE)

src = (src * 255).astype(np.uint8)
cv2.imshow('Output Image', src)
cv2.waitKey(0)
cv2.destroyAllWindows()




