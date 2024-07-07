import cv2
import numpy as np

def downsample(image, scale_factor):
    return cv2.resize(image, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_AREA)

def laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F)

def joint_bilateral_filter(image, guidance, d, sigma_color, sigma_space):
    return cv2.ximgproc.jointBilateralFilter(guidance, image, d, sigma_color, sigma_space)

def joint_bilateral_upsample(low_res, high_res_guidance, scale_factor, d, sigma_color, sigma_space):
    upsampled = cv2.resize(low_res, (high_res_guidance.shape[1], high_res_guidance.shape[0]), interpolation=cv2.INTER_LINEAR)
    return joint_bilateral_filter(upsampled, high_res_guidance, d, sigma_color, sigma_space)

# メイン処理
image = cv2.imread('asset/01.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ダウンサンプリング
downsampled = downsample(gray, 4)

# ラプラシアンフィルタ
laplacian = laplacian_filter(gray)

# ジョイントバイラテラルフィルタ
bilateral = joint_bilateral_filter(gray, image, d=9, sigma_color=75, sigma_space=75)

# ジョイントバイラテラルアップサンプリング
upsampled = joint_bilateral_upsample(downsampled, gray, 4, d=9, sigma_color=75, sigma_space=75)

# 結果の表示
cv2.imshow('Original', gray)
cv2.imshow('Downsampled', downsampled)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('Joint Bilateral', bilateral)
cv2.imshow('Joint Bilateral Upsampled', upsampled)
cv2.waitKey(0)
cv2.destroyAllWindows()