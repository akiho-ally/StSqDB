import pickle
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance



def main():
    # 1. movie_dicの読み込み
    with open("annotationed_movie.pkl", "rb") as annotationed_movie:
        movie_dic = pickle.load(annotationed_movie)

        # 1つのmovie_data = (images, labels)
        # data = [(images, labels), (images, labels), ..., (images, labels)]

    # 2. 前処理
    data = []
    for mid, frames in movie_dic.items():  ##frames:(filename, label_id, frame_id)
        images = []
        labels = []
        split_id = np.random.randint(1, 5)
        ##リサイズ
        for frame in frames:
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            img = Image.open(filepath)
            img_resize = np.array(img.resize((224, 224)))
            images.append(img_resize)
            labels.append(label_id)
            
            ##len(images):1467
        
        ##反転処理

        for frame in frames:  
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            img = Image.open(filepath)
            img_resize = np.array(img.resize((224, 224)))
            img_fliped = np.array(cv2.flip(img_resize, 1)) 
            images.append(img_fliped)
            labels.append(label_id)
        
             ##len(images):2934


        ##色補正

        for frame in frames:  
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            img = Image.open(filepath)
            img_resize = np.array(img.resize((224, 224)))
            img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
            img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
            img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
            images.append(img_bgr)
            labels.append(label_id)

            ##動画１本　len(images): 4401



        index = 0
        # split_images = []
        # split_labels = []
        for i in range(len(images)):
            if index+300 <=len(images):
                split_image = images[index : index+300]
                split_label = labels[index : index+300]
            else:
                break
            data.append((split_image, split_label, split_id))
            # split_images.append(split_image)
            # split_labels.append(split_label)

            index += 10
        



        # TODO: 系列長について、padding or 長い動画の分割
            # data.append((split_images, split_labels, split_id))  ##len(data)=1  imagesに反転画像も色補正画像もまとめて入れてしまったからだと思われる。。
                                                             ## 動画１本　len(split_images)=411   len(split_labels)=411
        print(mid)
    import pdb; pdb.set_trace()##len(data)=40
    ##動画の本数が増えたというより、１本の動画が長くなった感じ（動画３回繰り返したものを合わせたので動画１本）


    for i in range(1, 5):  
        ##評価
        images = []
        labels = []
        val_split = []
        train_split = []

        for movie_data in data:
            if movie_data[2] == i:
                val_split.append((movie_data[0], movie_data[1]))
            else:
                train_split.append((movie_data[0], movie_data[1]))
        # TODO: movieをシャッフル
  
        with open("val_split_{:1d}.pkl".format(i), "wb") as f:
            pickle.dump(val_split, f)
        with open("train_split_{:1d}.pkl".format(i), "wb") as f:
            pickle.dump(train_split, f)
        print("finish {}".format(i))

    print(data[0])
    
    # TODO: dataset 情報を表示


if __name__ == "__main__":
    main()