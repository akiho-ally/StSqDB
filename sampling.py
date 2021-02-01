import os
import pickle

from PIL import Image
import numpy as np
from natsort import natsorted
import collections

from itertools import groupby
from operator import itemgetter

import random

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

import itertools




element_names = ['Bracket', 'Change_edge', 'Chasse', 'Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle']



train_file_dict = {}
eval_file_dict = {}

img_id_list = []

seq_length  = 19
class_num = 2


for element_name in element_names:
    dir_path = '/home/akiho/projects/golfdb/data/dataset/train_all/' + element_name
    files_all = natsorted(os.listdir(dir_path))
    if  element_name =='Bracket' or element_name == 'Counter_turn' or element_name == 'Rocker_turn' or element_name== 'Three_turn' or element_name =='Twizzle' or element_name==  'Choctaw' or element_name== 'Mohawk':
        label_id = 0
    elif  element_name ==  'Chasse'  or element_name==  'Cross_roll'  or element_name == 'Toe_step' or element_name == 'Loop' or element_name=='Change_edge':
        label_id = 1
    # label_id = element_names.index(element_name)

    rename_files_all = []
    element_mid_dic = {}
    for file_name in files_all: ##.DSファイル除去
        if file_name.startswith("."):  #隠しファイルに対しては処理を行わない
            continue

        mid = file_name.split("_")[0].replace('img', '')
        file_num = file_name.split("_")[1].replace('.jpg', '')

        if mid == "" :
            mid_key = 1
            img1_file = file_name.replace('img','img1')
            # rename_files_all.append((mid_key,img1_file,file_num))
            rename_files_all.append(img1_file)
        else:
            mid_key = int(mid)
            # rename_files_all.append((mid_key,file_name,file_num))
            rename_files_all.append(file_name)

    files_all = natsorted(rename_files_all)



    for i in range(56):
        split_mid = [s for s in files_all if 'img' + str(i+1) + '_' in s]  ##midごとにリスト化

        if len(split_mid) == 0:
            continue
        else:
            next_id = int(split_mid[0].split("_")[1].replace('.jpg', ''))

            output = []
            for file in split_mid:
                img_id = int(file.split("_")[1].replace('.jpg', ''))

                if img_id == next_id:

                    output.append((file,label_id))



                else:

                    img_id_list.append(output)

                    output = []
                    output.append((file,label_id))


                next_id = img_id + 1
            img_id_list.append(output)



remove_img_id_list = []
for x in range(len(img_id_list)):

    if len(img_id_list[x]) <= seq_length-1 or len(img_id_list[x]) == 131:  ##12: 13フレーム以下のものを削除
        img_id_list[x].clear()

    remove_img_id_list.append(img_id_list[x])
remove_list = [x for x in remove_img_id_list if x]

sample_list = []

for x in range(len(remove_list)):
    sampling = natsorted(random.sample(remove_list[x], seq_length)) ##フレーム数を揃える
    sample_list.append(sampling)



############df作成
frames= []
labels = []
for a in range(len(sample_list)):
    frame = sample_list[a]
    element_id = sample_list[a][0][1]
    frames.append(frame)
    labels.append(element_id)


df=pd.DataFrame(frames,columns=['columns' + str(i) for i in range(seq_length)]) ##フレーム数分
df['label']=labels
print(df)

rs=RandomUnderSampler(random_state=42)
df_sample,_=rs.fit_sample(df,df.label)
print()
print('*'*20)
print('＜元のデータ＞')
for i in range(class_num):
    print(str(i) + 'の件数：%d'%len(df.query('label==' + str(i))))
print('*'*20)
print('＜アンダーサンプリング後のデータ＞')
for i in range(class_num):
    print(str(i) + 'の件数：%d'%len(df_sample.query('label==' + str(i))))

# import pdb; pdb.set_trace()


#######train, testに分類

X_train = []
X_test = []
for i in range(class_num):
    train_label = df_sample.query('label==' + str(i))[:164]
    train = train_label.iloc[:, 0:seq_length].values.tolist()
    X_train.append(train)

    test_label = df_sample.query('label==' + str(i))[164:205]
    test = test_label.iloc[:, 0:seq_length].values.tolist()
    X_test.append(test)

X_train = list(itertools.chain.from_iterable(X_train))
X_test = list(itertools.chain.from_iterable(X_test))



print('len(X_train) : ' + str(len(X_train)))
print('len(X_test) : ' + str(len(X_test)))

with open("anno_data_train_sampling.pkl", "wb") as anno_data_t:
    pickle.dump(X_train, anno_data_t)

with open("anno_data_eval_sampling.pkl", "wb") as anno_data_e:
    pickle.dump(X_test, anno_data_e)










##class_num 12 seq_length 13, 19
# ＜アンダーサンプリング後のデータ＞
# 0の件数：15
# 1の件数：15
# 2の件数：15
# 3の件数：15
# 4の件数：15
# 5の件数：15
# 6の件数：15
# 7の件数：15
# 8の件数：15
# 9の件数：15
# 10の件数：15
# 11の件数：15

# ##class_num 2 seq_length 19 turn or step
# ＜アンダーサンプリング後のデータ＞
# 0の件数：205
# 1の件数：205
# len(X_train) : 328
# len(X_test) : 82












#############
