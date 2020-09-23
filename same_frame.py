import pickle
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', default=300)
    parser.add_argument('--img_size', default=224) 
    parser.add_argument('--use_no_element', action='store_true')
    args = parser.parse_args() 

    with open("anno_data_train.pkl", "rb") as annotationed_train:
        anno_element_train = pickle.load(annotationed_train) 
    
    with open("anno_data_eval.pkl", "rb") as annotationed_eval:
        anno_element_eval = pickle.load(annotationed_eval) 
    
    element_names = ['Bracket', 'Change_edge', 'Chasse', 'Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','No_element']
    


    train_data = []
    for file_name, element_name in anno_element_train.items():
        images = []
        labels = []
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        images.append(img_resize)
        label_id = element_names.index(element_name)
        labels.append(label_id)

        train_data.append((images, labels))


    for file_name, element_name in anno_element_train.items():
        images = []
        labels = []
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        img_fliped = np.array(cv2.flip(img_resize, 1)) 
        images.append(img_fliped)
        label_id = element_names.index(element_name)
        labels.append(label_id)

        train_data.append((images, labels))


    for file_name, element_name in anno_element_train.items():
        images = []
        labels = []
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
        img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
        img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        images.append(img_bgr)
        label_id = element_names.index(element_name)
        labels.append(label_id)

        train_data.append((images, labels))


    with open("data/same_frames/train_split_1.pkl", "wb") as f:
            pickle.dump(train_data, f)
    











    eval_data = []
    for file_name, element_name in anno_element_eval.items():
        eval_data = []
        images = []
        labels = []
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        images.append(img_resize)
        label_id = element_names.index(element_name)
        labels.append(label_id)

        eval_data.append((images, labels))
    
    for file_name, element_name in anno_element_eval.items():
        eval_data = []
        images = []
        labels = []
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        img_fliped = np.array(cv2.flip(img_resize, 1)) 
        images.append(img_fliped)
        label_id = element_names.index(element_name)
        labels.append(label_id)

        eval_data.append((images, labels))

    for file_name, element_name in anno_element_eval.items():
        eval_data = []
        images = []
        labels = []
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
        img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
        img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        images.append(img_bgr)
        label_id = element_names.index(element_name)
        labels.append(label_id)

        eval_data.append((images, labels))

    with open("data/same_frames/val_split_1.pkl", "wb") as f:
            pickle.dump(eval_data, f)



if __name__ == "__main__":
    main()