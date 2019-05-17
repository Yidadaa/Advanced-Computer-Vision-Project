import numpy as np

"""
图片滤波器

自动padding，输出卷积后的图像
"""
def imfilter(im:list, kernel:list, step=1)->list:
    src_img = np.array(im)
    kernel = np.array(kernel)

    im = src_img

    # 将灰度图与彩色图像保持一致
    if len(im.shape) == 2:
        im = im.reshape(im.shape + (1, ))

    # 分别取图像宽高和滤波器宽高
    [ih, iw] = im.shape[0:2]
    [kh, kw] = kernel.shape[0:2]
    
    # 不合法的输入
    if iw < kw or ih < kh:
        return None

    # 计算padding值
    wp = getPad(kw - 1)
    hp = getPad(kh - 1)

    # padding
    pimg = np.pad(im, (hp, wp, (0, 0)), 'constant', constant_values=(0, 0))

    # 变换，将像素拆分成R, G, B各通道的图片
    pimg = pimg.transpose()

    # 设定输出值，输出矩阵与输入矩阵保持一致
    ret_img = np.zeros(im.shape[::-1])

    # 获取padding后的图像宽高
    [channel, ph, pw] = pimg.shape

    # 执行滤波，对每一个通道都进行处理
    for chn in range(channel):
        for r in range(0, ph - kh):
            for c in range(0, pw - kw):
                ret_img[chn, r, c] = np.sum(pimg[chn, r:r + kh, c:c + kw] * kernel)

    # 结果转置为RGB图像
    ret_img = ret_img.transpose()
    if ret_img.shape[-1] == 1:
        ret_img = ret_img.reshape(ret_img.shape[0:2])

    return ret_img.astype(np.uint8)

def getPad(dis:int)->tuple:
    dis = 0 if dis < 0 else dis
    return ((dis - dis % 2) // 2, (dis + dis % 2) // 2)

if __name__ == "__main__":
    print('测试函数')
    test_img = np.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ])

    test_img_2 = [
        [[1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3]]
    ]

    kernel = np.array([
        [1, 1],
        [1, 1]
    ])

    kernel = kernel / np.sum(kernel)

    for img in [test_img, test_img_2]:
        print('Test Img:\n ', img)
        print('Test kernel:\n', kernel)

        imfilter(img, kernel)
        print('\n')