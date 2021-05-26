import imageio
import numpy as np
import cv2
import time
import glob
import os
import argparse
#mask, marker, result = imageio.imread('mask.png'),imageio.imread('marker.png'), imageio.imread('result.png')
#image = 255 - mask[1:-1, 1:-1]
def imfill(image):
    mask = np.pad(image, 1, 'constant',  constant_values=0)
    mask = 255 - mask
    marker = mask.copy()
    h, w = marker.shape[0]-1, marker.shape[1]-1
    marker[1:h, 1:w] = 0
    # mask[1:h, 1:w] = 255 - mask[1:h, 1:w]
    queue = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            marker[i, j] = min(max(marker[i-1, j], marker[i, j-1], marker[i, j]), mask[i, j])

    for i in range(h-1, 0, -1):
        for j in range(w-1, 0, -1):
            marker[i, j] = min(max(marker[i+1, j], marker[i, j+1],  marker[i, j]), mask[i, j])
            if (marker[i+1, j] < marker[i, j] and marker[i+1, j] < mask[i+1, j] ) or  (marker[i, j+1] < marker[i, j] and marker[i, j+1] < mask[i, j+1]):
                queue.append((i, j))

    while queue:
        i, j = queue.pop(0)
        for (x, y) in [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]:
            if marker[x, y] < marker[i, j] and marker[x, y] != mask[x, y]:
                marker[x, y] = min(marker[i, j], mask[x, y])
                queue.append((x, y))

    res = (255 - marker[1:-1, 1:-1])
    compare = np.clip(((255 - marker[1:-1, 1:-1]).astype(np.float32)) - image.astype(np.float32) , 0, 255).astype(np.uint8)

    return res, compare


def denoiser(input_path, output_path, debug):
    mask = imageio.imread('./img_07170.png').astype(np.float32)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    holefillMask = cv2.imread('./result-mask.png', -1)
    if debug:
        img = imageio.imread('./input.png')
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
        cv2.imwrite('./result.png', img)
        print('success')
        return 
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
    parser.add_argument("--debug", type =  int, default= 0, help = 'debug')
    argparse = parser.parse_args()
    denoiser(argparse.input_path, argparse.output_path, argparse.debug)
# for k in range(len(points[0])):
#         x = points[0][k]
#         y = points[1][k]
#         if (x + 32) >= 2160 or (y + 32) >= 3840:
#             continue
#         temp = np.mean(filterd_y[x:x + 32, y: y + 32]).astype(np.uint8)
#         if temp < 40 and not visited[x, y]:
#             #holefillMask[x: x + 32, y: y + 32] = holefillMask[x: x + 32, y: y + 32] - filterd_y[x: x + 32, y: y + 32]
#             filterd_y[x: x + 32, y: y + 32], holefillMask[x: x + 32, y: y + 32] = imfill(filterd_y[x: x + 32, y: y + 32])
#             visited[x:x + 8, y:y + 8] = 1



#cv2.imwrite('./holefillMask.png', holefillMask)


# mask = imageio.imread('E:/settlelite/mask_y.png')
#
# img = imageio.imread('E:/settlelite/hole_fill_result/img_00001.png')
# yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
# y_img = yuv[:,:, 0]
# points = np.where(mask > 10 )
# visited = np.zeros(())
# for j in range(len(points[0])):
#     x = points[0][j]
#     y = points[1][j]
#
#     if x > 0 and y > 0 and y_img[x, y] < 80 :
#         y_img[x, y] = np.mean([y_img[x - 1, y], y_img[x - 1, y - 1], y_img[x, y - 1]])
#
#
#
# points = np.where(mask > 20)
# visited = np.zeros((mask.shape[0], mask.shape[1]))
# start = time.time()
# for k in range(len(points[0])):
#     x = points[0][k]
#     y = points[1][k]
#     if (x + 32) >= 2160 or (y + 32) >= 3840:
#         continue
#
#     temp = np.mean(y_img[x:x + 32, y: y + 32]).astype(np.uint8)
#     if temp < 40 and not visited[x, y]:
#         y_img[x: x + 32, y: y + 32] = imfill(y_img[x: x + 32, y: y + 32])
#         visited[x:x+8, y:y+8] = 1
# end = time.time()
# print(end - start)
# yuv[:,:, 0] = y_img
# img = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
# cv2.imwrite('E:/settlelite/imfill/mask_hole.png', img)


