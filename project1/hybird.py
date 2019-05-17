import cv2
import numpy as np
import matplotlib.pyplot as plt

from imfilter import imfilter

FILE_PATH = __file__.rsplit('/', 1)[0] + '/img/'

def imread(filename):
    rgb_img = cv2.imread(FILE_PATH + filename)
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

def gaussianKernel(sigma, n=2):
    size = n * sigma + 1
    kernel = np.zeros((size, size))
    center = size / 2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-0.5 * (x * x + y * y) / 2 / sigma)
    return kernel / np.sum(kernel)

'''
测试滤波器
'''
def testFilter():
    rgb_img = imread('lena.jpeg')
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    size = 3
    kernel = np.ones((size, size)) / size / size

    rgb_img_filtered = imfilter(rgb_img, kernel)
    gray_img_filtered = imfilter(gray_img, kernel)

    cv2.imwrite(FILE_PATH + '/out.jpg', gray_img_filtered)

    high_pass_img = rgb_img - rgb_img_filtered

    plt.figure(figsize=(12, 6))
    imgs = [rgb_img, gray_img, rgb_img_filtered, gray_img_filtered, high_pass_img]
    for index, img in enumerate(imgs):
        plt.subplot(1, len(imgs), index + 1)
        cmap = 'gray' if len(img.shape) == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])

    plt.show()

def hybird():
    cat_img = imread('cat.jpg')
    dog_img = imread('dog.jpg')

    cat_kernel = gaussianKernel(3)
    dog_kernel = gaussianKernel(2)

    cat_blur = imfilter(cat_img, cat_kernel)
    dog_blur = imfilter(dog_img, dog_kernel)

    # 需要临时扩展
    cat_high = cat_img.astype(np.int16) - cat_blur.astype(np.int16)

    hybird_img = cat_high + dog_blur
    plt.figure()
    plt.imshow(hybird_img)
    plt.show()

if __name__ == "__main__":
    # testFilter()
    hybird()