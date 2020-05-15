import cv2
import numpy as np 
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions 
def simple_add(base_img, mask, beta):
        base_img = base_img.astype(np.float32)
        mask = mask.astype(np.float32)
        resize_mask = cv2.resize(mask, (base_img.shape[1], base_img.shape[0]))
        # print(type(base_img[0,0,0]), type(resize_mask[0,0,0]))
        # tmp = np.where(resize_mask >= 10, 1.0, 0.0)
        # reversed_tmp = 1 - tmp
        # c = tmp * resize_mask + reversed_tmp * base_img
        c = cv2.addWeighted(base_img, 1 , resize_mask, beta , 0)
        return c
    
def extract_img(vid_path):
    vidcap = cv2.VideoCapture(vid_path)
    success,image = vidcap.read()
    count = 0
    imgs = []
    while success:
        imgs.append(image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    return imgs
if __name__ == "__main__":
    video_name = "./video/1.mp4"
    imgs = extract_img(video_name)
    model = ResNet50(weights='imagenet')
    redundancy_path = './elements/light_2.jpeg'
    redundancy = cv2.imread(redundancy_path)
    for i in range(len(imgs)):
        test_img = imgs[i]
        test_img = cv2.resize(test_img, (224,224))
        test_img = simple_add(test_img,redundancy, 1.0)
        test_img = np.expand_dims(test_img, axis = 0)
        test_img = preprocess_input(test_img)
        preds = model.predict(test_img)
        print("Predictions:", decode_predictions(preds, top=3)[0] )
