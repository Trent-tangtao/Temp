from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import os
from keras.applications.resnet50 import preprocess_input
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator


# 载入模型
def read_model():
    model = load_model('/Users/tang/Desktop/imageclass/model.h5')
    return model



# 测试数据集读取
def read_test(test_data_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    model = load_model('/Users/tang/Desktop/imageclass/model.h5')
    score = model.evaluate_generator(test_generator, steps=1)
    print("样本准确率%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    # y = model.evaluate_generator(test_generator, 20, max_q_size=10,workers=1, use_multiprocessing=False)
    # name_list = model.predict_generator.filenames()
    # print(name_list)
    # return y


if __name__ == '__main__':
    img_dir = './data/test'
    read_test(img_dir)   # 测试集验证,批量
