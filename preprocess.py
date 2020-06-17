import pickle
import numpy as np
from PIL import Image

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
        for frame in frames:
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            img = Image.open(filepath)
            img_resize = np.array(img.resize((224, 224)))
            images.append(img_resize)
            labels.append(label_id)

        # TODO: 系列長について、padding or 長い動画の分割
        data.append((images[:300], labels[:300], split_id))

        print(mid)

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