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
***Average PCE: 0.5219493844049248
Bracket: 0.03172192693832785 (752.0 / 23706.0)
Change_edge: 0.45584415584415583 (1755.0 / 3850.0)
Chasse: 0.0 (0.0 / 2610.0)
Choctaw: 0.05675693883050391 (1685.0 / 29688.0)
Counter_turn: 0.034984513688236636 (1683.0 / 48107.0)
Cross_roll: 0.0 (0.0 / 4287.0)
Loop: 0.1385322671361847 (5665.0 / 40893.0)
Mohawk: 0.05349938607261884 (915.0 / 17103.0)
Rocker_turn: 0.049245137838626656 (1856.0 / 37689.0)
Three_turn: 0.004918433433127635 (161.0 / 32734.0)
Toe_step: 0.0018832391713747645 (22.0 / 11682.0)
Twizzle: 0.409170074208693 (21614.0 / 52824.0)
No_element: 0.7372833799803156 (421746.0 / 572027.0)
iteration=3000 seq=300 bs=8***


## 今やってること
### pallas
iteration=3000, bs=8, seq=300 
### glacus
iteration=8000, bs=4, seq=500 

