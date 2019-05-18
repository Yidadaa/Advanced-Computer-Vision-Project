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
            kernel[x, y] = np.exp(-0.5 * (x * x + y * y) / 2 / sigma / sigma)
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

    plt.figure(figsize=(12, 4))
    imgs = [rgb_img, gray_img, rgb_img_filtered, gray_img_filtered]
    for index, img in enumerate(imgs):
        plt.subplot(1, len(imgs), index + 1)
        cmap = 'gray' if len(img.shape) == 2 else None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.title('(%d)' % (index + 1))

    plt.savefig(FILE_PATH + 'filter.pdf')

def hybird():
    cat_img = imread('cat.jpg')
    dog_img = imread('dog.jpg')

    cat_kernel = gaussianKernel(3)
    dog_kernel = gaussianKernel(2)

    cat_blur = imfilter(cat_img, cat_kernel)
    dog_blur = imfilter(dog_img, dog_kernel)

    # 需要临时扩展
    cat_high = cat_img.astype(np.int16) - cat_blur.astype(np.int16)
    cat_highpass = (cat_high + 128).astype(np.uint8)

    hybird_img = cat_high + dog_blur
    plt.figure(figsize=(12, 6))

    imgs = [cat_img, dog_img, cat_blur, dog_blur, cat_highpass, hybird_img]
    for index, img in enumerate(imgs):
        plt.subplot(2, len(imgs) // 2, index + 1)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('(%d)' % (index + 1))

    plt.savefig(FILE_PATH + 'hybird.pdf')

if __name__ == "__main__":
    testFilter()
    hybird()