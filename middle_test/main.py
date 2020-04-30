import run
import cv2

if __name__ == '__main__':
    # 读入图片
    img = cv2.imread('../image/lena-gray.png', -1)
    # 显示结果
    run.run(img)