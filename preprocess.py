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
        with open("anno_data_train.pkl", "rb") as annotationed_movie:
            frame_dic = pickle.load(annotationed_movie)
        print('train')
    elif args.eval == True:
        with open("anno_data_eval.pkl", "rb") as annotationed_movie:
            frame_dic = pickle.load(annotationed_movie)
        print('eval')

        # 1つのmovie_data = (images, labels)
        # data = [(images, labels), (images, labels), ..., (images, labels)]

    # 2. 前処理
    data = []


    images = []
    labels = []
    for frame, ml_id in frame_dic.items():  ##frames:(filename, (m,id, label_id,)
        ##############リサイズ

        filename = frame
        label_id = ml_id[1]
        mid = ml_id[0]
        filepath = "data/videos_56/img" +str( mid )+ '/' + filename
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        images.append(img_resize)
        labels.append(label_id)




        #len(images):1467


    index = 0
    for i in range(len(images)):
        if index+int(args.seq_length) <=len(images) + 1:

            split_image = images[index : index+int(args.seq_length) ]
            split_label = labels[index : index+int(args.seq_length) ]
        else:
            break
        data.append((split_image, split_label))  ##split_id = 3
        index += int(args.seq_length)
    # data.append((images, labels, split_id))
    print('len(data) : ' + str(len(data)))
    import pdb; pdb.set_trace()

    ################反転処理
    fliped_images = []
    fliped_labels = []
    for frame, ml_id in frame_dic.items():
        filename = frame
        label_id = ml_id[1]
        mid = ml_id[0]
        filepath = "data/videos_56/img" +str( mid )+ '/' + filename
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        img_fliped = np.array(cv2.flip(img_resize, 1))
        fliped_images.append(img_fliped)
        fliped_labels.append(label_id)



    index = 0

    for i in range(len(fliped_images)):
        if index+int(args.seq_length) <=len(images) + 1:
            split_fliped_image = fliped_images[index : index+int(args.seq_length)]
            split_fliped_label = fliped_labels[index : index+int(args.seq_length) ]
        else:
            break
        data.append((split_fliped_image, split_fliped_label))
        index += int(args.seq_length)

    # import pdb; pdb.set_trace() ##len(data) = 234
    # data.append((fliped_images, fliped_labels, split_id))

    print('len(data) : ' + str(len(data)))
    import pdb; pdb.set_trace()

    ##############色補正
    bgr_images = []
    bgr_labels = []
    for frame, ml_id in frame_dic.items():
        filename = frame
        label_id = ml_id[1]
        mid = ml_id[0]
        filepath = "data/videos_56/img" +str( mid )+ '/' + filename
        img = Image.open(filepath)
        img_resize = np.array(img.resize((args.img_size, args.img_size)))
        img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
        img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
        img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        bgr_images.append(img_bgr)
        bgr_labels.append(label_id)




    index = 0
    for i in range(len(bgr_images)):
        if index+int(args.seq_length) <=len(images) + 1:
            split_bgr_image = bgr_images[index : index+int(args.seq_length)]
            split_bgr_label = bgr_labels[index : index+int(args.seq_length )]
        else:
            break
        data.append((split_bgr_image, split_bgr_label))
        index += int(args.seq_length)

    print('len(data) : ' + str(len(data)))
    import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()  ##len(data)=351

    # data.append((bgr_images, bgr_labels, split_id))


    # TODO: 系列長について、padding or 長い動画の分割
        # data.append((split_images, split_labels, split_id))


    # import pdb; pdb.set_trace() ##len(data)=10197


    if args.train == True:
        with open("data/sameframes/seq_length_{}/train_split_1.pkl".format(args.seq_length), "wb") as f:
            pickle.dump(data, f)
    elif args.eval == True:
        with open("data/sameframes/seq_length_{}/val_split_1.pkl".format(args.seq_length), "wb") as f:
            pickle.dump(data, f)


    # TODO: dataset 情報を表示


if __name__ == "__main__":
    main()