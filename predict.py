'''
Author: [egrt]
Date: 2022-08-23 13:21:27
LastEditors: [egrt]
LastEditTime: 2022-08-23 13:22:19
Description: 
'''
#--------------------------------------------------------------#
#   对单张图片进行预测，运行结果保存在根目录
#   默认保存文件为results/predict_out/predict_srgan.png
#--------------------------------------------------------------#
from PIL import Image

from HEAT import HEAT

if __name__ == "__main__":
    heat = HEAT()
    #----------------------------#
    #   单张图片的保存路径
    #----------------------------#
    save_path = "results/predict_out/predict_maskgan.png"

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = heat.detect_one_image(image)
            r_image.save(save_path)
            r_image.show()