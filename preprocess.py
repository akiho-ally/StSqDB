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
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    # 1. movie_dicの読み込み
    # if args.use_no_element == False:
    #     with open("annotationed_movie_12.pkl", "rb") as annotationed_movie:
    #         movie_dic = pickle.load(annotationed_movie)
    # else:
    #     with open("annotationed_movie.pkl", "rb") as annotationed_movie:
    #         movie_dic = pickle.load(annotationed_movie)
    if args.train == True:
        with open("anno_data_train_sampling.pkl", "rb") as annotationed_movie:
            frame_dic = pickle.load(annotationed_movie)
        print('train')
    elif args.eval == True:
        with open("anno_data_eval_sampling.pkl", "rb") as annotationed_movie:
            frame_dic = pickle.load(annotationed_movie)
        print('eval')

        # 1つのmovie_data = (images, labels)
        # data = [(images, labels), (images, labels), ..., (images, labels)]

    # 2. 前処理



    if args.train == True:
        data = []
        for mv in frame_dic: ##mv : [('img10_93.jpg', 3), ('img10_94.jpg', 3), ('img10_95.jpg', 3), ('img10_96.jpg', 3), ('img10_97.jpg',
            ##############リサイズ
            images = []
            labels = []
            fliped_images = []
            fliped_labels = []
            bgr_images = []
            bgr_labels = []
            for frame_label in mv:
                filename = frame_label[0]
                label_id =  frame_label[1]
                mid = filename.split("_")[0]
                if mid == 'img1':
                    filename = filename.replace("img1", "img")
                    filepath =  "data/videos_56/img1/" + filename
                else:
                    filepath = "data/videos_56/" +str( mid )+ '/' + filename
                img = Image.open(filepath)
                img_resize = np.array(img.resize((args.img_size, args.img_size)))
                images.append(img_resize)
                labels.append(label_id)


                img_fliped = np.array(cv2.flip(img_resize, 1))
                fliped_images.append(img_fliped)
                fliped_labels.append(label_id)

                img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
                img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
                img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
                bgr_images.append(img_bgr)
                bgr_labels.append(label_id)

            data.append((images, labels))
            data.append((fliped_images, fliped_labels))
            data.append((bgr_images, bgr_labels))
        print('train : len(data) : ' + str(len(data)))

        with open("data/sampling/two/turn_step/seq_length_{}/train_split_1.pkl".format(args.seq_length), "wb") as f:
            pickle.dump(data, f)

    elif args.eval == True:
        data = []
        for mv in frame_dic: ##mv : [('img10_93.jpg', 3), ('img10_94.jpg', 3), ('img10_95.jpg', 3), ('img10_96.jpg', 3), ('img10_97.
            ##############リサイズ
            images = []
            labels = []
            fliped_images = []
            fliped_labels = []
            bgr_images = []
            bgr_labels = []
            for frame_label in mv:
                filename = frame_label[0]
                label_id =  frame_label[1]
                mid = filename.split("_")[0]
                if mid == 'img1':
                    filename = filename.replace("img1", "img")
                    filepath =  "data/videos_56/img1/" + filename
                else:
                    filepath = "data/videos_56/" +str( mid )+ '/' + filename
                img = Image.open(filepath)
                img_resize = np.array(img.resize((args.img_size, args.img_size)))
                images.append(img_resize)
                labels.append(label_id)

                img_fliped = np.array(cv2.flip(img_resize, 1))
                fliped_images.append(img_fliped)
                fliped_labels.append(label_id)

                img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
                img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
                img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
                bgr_images.append(img_bgr)
                bgr_labels.append(label_id)

            data.append((images, labels))
            data.append((fliped_images, fliped_labels))
            data.append((bgr_images, bgr_labels))
        print('eval : len(data) : ' + str(len(data)))




        with open("data/sampling/two/turn_step/seq_length_{}/val_split_1.pkl".format(args.seq_length), "wb") as f:
            pickle.dump(data, f)


    # TODO: dataset 情報を表示


if __name__ == "__main__":
    main()