import pickle
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', default=150)
    parser.add_argument('--img_size', default=224) 
    parser.add_argument('--use_no_element', action='store_true')
    args = parser.parse_args() 

    with open("anno_data_train.pkl", "rb") as annotationed_train:
        anno_element_train = pickle.load(annotationed_train) 
    
    with open("anno_data_eval.pkl", "rb") as annotationed_eval:
        anno_element_eval = pickle.load(annotationed_eval) 


    element_names = ['Bracket', 'Change_edge', 'Chasse', 'Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','No_element']



 
    train_data = []
    images = []
    labels = []
    for file_name, element_name in anno_element_train.items():

        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        images.append(img_resize)
        label_id = element_names.index(element_name)
        labels.append(label_id)


    index = 0
    for i in range(len(images)):
        if index+30 <=len(images):
            split_image = images[index : index+int(30)]
            split_label = labels[index : index+int(30)]
            train_data.append((split_image, split_label))
            index += 30 
        else:
            break
        
    fliped_images = []
    fliped_labels = []
    for file_name, element_name in anno_element_train.items():

        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        label_id = element_names.index(element_name)
        img_fliped = np.array(cv2.flip(img_resize, 1)) 
        fliped_images.append(img_fliped)
        fliped_labels.append(label_id)


    index = 0
    for i in range(len(images)):
        if index+30 <=len(images):
            split_image = fliped_images[index : index+int(30)]
            split_label = fliped_labels[index : index+int(30)]
            train_data.append((split_image, split_label))
            index += 30 
        else:
            break
    
    bgr_images = []
    bgr_labels = []
    for file_name, element_name in anno_element_train.items():

        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        label_id = element_names.index(element_name)
        img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
        img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
        img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        bgr_images.append(img_bgr)
        bgr_labels.append(label_id)


    index = 0
    for i in range(len(images)):
        if index+30 <=len(images):
            split_image = bgr_images[index : index+int(30)]
            split_label = bgr_labels[index : index+int(30)]
            train_data.append((split_image, split_label))
            index += 30
        else:
            break
        
    import pdb; pdb.set_trace()





    with open("data/same_frames/train_split_1.pkl", "wb") as f:
            pickle.dump(train_data, f)
    











    eval_data = []
    images = []
    labels= []
    for file_name, element_name in anno_element_eval.items():

        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        images.append(img_resize)
        label_id = element_names.index(element_name)
        labels.append(label_id)



    index = 0
    for i in range(len(images)):
        if index+int(30) <=len(images):
            split_image = images[index : index+int(30)]
            split_label = labels[index : index+int(30)]
        else:
            break
        eval_data.append((split_image, split_label))  ##split_id = 3
        index += 30



    fliped_images = []
    fliped_labels = []
    for file_name, element_name in anno_element_eval.items():

        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        label_id = element_names.index(element_name)
        img_fliped = np.array(cv2.flip(img_resize, 1)) 
        fliped_images.append(img_fliped)
        fliped_labels.append(label_id)



    index = 0
    for i in range(len(images)):
        if index+int(30) <=len(images):
            split_image = fliped_images[index : index+int(30)]
            split_label = fliped_labels[index : index+int(30)]
        else:
            break
        eval_data.append((split_image, split_label))  ##split_id = 3
        index += 30


    bgr_images = []
    bgr_labels = []
    for file_name, element_name in anno_element_eval.items():

        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        label_id = element_names.index(element_name)
        img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
        img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
        img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        bgr_images.append(img_bgr)
        bgr_labels.append(label_id)



    index = 0
    for i in range(len(images)):
        if index+int(30) <=len(images):
            split_image = bgr_images[index : index+int(30)]
            split_label = bgr_labels[index : index+int(30)]
        else:
            break
        eval_data.append((split_image, split_label))  ##split_id = 3
        index += 30
    import pdb; pdb.set_trace()



    with open("data/same_frames/val_split_1.pkl", "wb") as f:
            pickle.dump(eval_data, f)



if __name__ == "__main__":
    main()