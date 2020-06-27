# StSqDB: A Video Database for Figure Skating Step Sequencing


## Getting Started
1 anno_data.py
 * imageとlavel(0~12)の情報が入っているpklファイルの作成

2 preprocess.py
 * DataAugmentationを行う  
  * リサイズ(224*224),反転,色補正  
  * 指定のフレーム数ごとに一つの動画にする  
  * 交差検証のために4つにsplit(pklファイルの作成)


## Train
iteration,bs,seq_lengthを変えて実験中



## Evaluate
エレメントごとに精度を出したい


## これまでの精度  
Average PCE: 0.34063427299703264  
 ・iteration=8000 bs=16, seq=100  
***Average PCE: 0.5135715914272686  
 ・iteration=8000 bs=8, seq=300***      
Average PCE: 0.4110422848664688  
 ・iteration=6000, bs=16 seq=100  
Average PCE: 0.18964634146341464  
 ・iteration=6000, bs= 32 seq=50  
***Average PCE: 0.5131084967320262  
 ・iteration=8000 seq=500 bs=4***


## 今やってること
### pallas
iteration=8000, bs=4, seq=300 eval(bs=2)
### glacus
iteration=8000, bs=4, seq=500 でeval

