import imageio
import numpy as np
import cv2
import time
import glob
import os
import argparse
#mask, marker, result = imageio.imread('mask.png'),imageio.imread('marker.png'), imageio.imread('result.png')
#image = 255 - mask[1:-1, 1:-1]

def denoiser(input_path, output_path, debug):
    mask = imageio.imread('./img_07170.png').astype(np.float32)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    holefillMask = cv2.imread('./result-mask.png', -1)
    files = glob.glob(input_path + '/*.png')
    for file in files:
        img = imageio.imread(file).astype(np.float32)
        img = np.clip(img - mask, 0, 255).astype(np.uint8)

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        filterd_u = cv2.medianBlur(yuv[:, :, 1], 7)
        filterd_u = cv2.medianBlur(filterd_u, 7)

        filterd_v = cv2.medianBlur(yuv[:, :, 2], 7)
        filterd_v = cv2.medianBlur(filterd_v, 7)

        filterd_y = cv2.medianBlur(yuv[:, :, 0], 3)

        yuv[:, :, 1] = filterd_u
        yuv[:, :, 2] = filterd_v

        points = np.where(gray_mask > 10)
        for j in range(len(points[0])):
                x = points[0][j]
                y = points[1][j]
                if x > 0 and y > 0 and filterd_y[x, y] < 80:
                    filterd_y[x, y] = np.mean([filterd_y[x - 1, y], filterd_y[x - 1, y - 1], filterd_y[x, y - 1]])

        yuv[:, :, 0] = filterd_y + holefillMask
        img = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)

        cv2.imwrite(output_path + './{}'.format(os.path.basename(file)), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise for JP satellite")
    parser.add_argument("--input_path", type = str, default = './', help = 'path to the input noisy frames.')
    parser.add_argument("--output_path", type =  str, default= ' ./', help = 'path to save the denoised frames.')
    argparse = parser.parse_args()
    denoiser(argparse.input_path, argparse.output_path)


