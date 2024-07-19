import cv2
import numpy as np

# pyrDownだと0.5倍で固定のため自分で実装
def downsample(src, scale=0.5):
    height, width = src.shape[:2]
    height, width = int(height * scale), int(width * scale)
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)

# srcをtargetの大きさに合わせる
def upsample(src, target):
    height, width = target.shape[:2]
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)

# 画像のパス
path = "02.png"
image_path = "input" + "/" + path

# 画像の読み込み
image = cv2.imread(image_path)
image = image.astype(np.float32) / 255.0

# ピラミッドのレベル数
n = 20

# 画像のコピーを作成
G = image.copy()
gpA = [G]

scale = 0.8
# Gaussianピラミッドの作成
for i in range(n):
    G = downsample(G,scale)
    gpA.append(G)

# Laplacianピラミッドの作成
lpA = []
for i in range(n):
    GE = upsample(gpA[i+1], gpA[i]) #gpA[i+1]をgpA[i]のサイズに拡大
    GE = cv2.resize(GE, (gpA[i].shape[1], gpA[i].shape[0]))  # サイズを合わせる
    L = cv2.subtract(gpA[i], GE)
    lpA.append(L)

sigmaColor = 0.09
sigmaSpace = 20
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
cv2.imwrite("output" + "/" + path, src)
