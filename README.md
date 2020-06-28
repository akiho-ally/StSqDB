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
Average PCE: 0.45616735066119474  
Bracket: 0.013962709862482072  
Change_edge: 0.0  
Chasse: 0.0  
Choctaw: 0.3495014820803018  
Counter_turn: 0.16201384413910658  
Cross_roll: 0.0  
Loop: 0.008901278947497126  
Mohawk: 0.0  
Rocker_turn: 0.04001167449388416  
Three_turn: 0.042524592167165635  
Toe_step: 0.19012155452833418  
Twizzle: 0.21124867484476753  
No_element: 0.6380905097137023  
 ・ iteration=8000 bs=4 seq=300  


## 今やってること
### pallas
iteration=8000, bs=4, seq=300 eval(bs=2)
### glacus
iteration=8000, bs=4, seq=500 でeval

