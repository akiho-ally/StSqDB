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
 Average PCE: 0.48395074224021595  
Bracket: 0.10578182498596644 1696.0/16033.0  
Change_edge: 0.11934156378600823 1015.0/8505.0  
Chasse: 0.0 0.0/2610.0  
Choctaw: 0.14767465683261627 3604.0/24405.0  
Counter_turn: 0.08133971291866028 3383.0/41591.0  
Cross_roll: 0.0 0.0/2241.0  
Loop: 0.326181267866151 9699.0/29735.0  
Mohawk: 0.04312569111684482 351.0/8139.0  
Rocker_turn: 0.11468505576896376 2478.0/21607.0  
Three_turn: 0.12718231446948813 2353.0/18501.0  
Toe_step: 7.527853056308341e-05 1.0/13284.0  
Twizzle: 0.2609820658477828 8775.0/33623.0  
No_element: 0.6805726311720525 253531.0/372526.0  
iteration =8000 split=2 seq = 300 bs =8   
Average PCE: 0.5051728395061729  
Bracket: 0.1390697235588263 2948.0/21198.0  
Change_edge: 0.02242493348536678 177.0/7893.0  
Chasse: 0.0 0.0/2070.0  
Choctaw: 0.10572778729812536 2036.0/19257.0  
Counter_turn: 0.1498603418075084 6331.0/42246.0  
Cross_roll: 0.0 0.0/5676.0  
Loop: 0.07740505108926161 2409.0/31122.0  
Mohawk: 0.0 0.0/2493.0  
Rocker_turn: 0.08671803927864291 3241.0/37374.0  
Three_turn: 0.008884838819911152 234.0/26337.0  
Toe_step: 0.023520084566596194 267.0/11352.0  
Twizzle: 0.08037455352833285 4163.0/51795.0  
No_element: 0.7850878883415942 305546.0/389187.0  
iteration=8000 bs =8 seq=300 split 3  
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
Average PCE: 0.18536309523809524  
Bracket: 0.18741058655221746 917.0/4893.0  
Change_edge: 0.08464912280701754 193.0/2280.0  
Chasse: 0.0 0.0/435.0  
Choctaw: 0.11613876319758673 693.0/5967.0  
Counter_turn: 0.06308037143772013 591.0/9369.0  
Cross_roll: 0.0 0.0/1260.0  
Loop: 0.40485883125410377 3083.0/7615.0  
Mohawk: 0.10284552845528455 253.0/2460.0  
Rocker_turn: 0.18958592992660298 1369.0/7221.0  
Three_turn: 0.08398100172711571 389.0/4632.0  
Toe_step: 0.0013157894736842105 3.0/2280.0  
Twizzle: 0.44803895990955733 5152.0/11499.0  
No_element: 0.17113674842028329 18498.0/108089.0  
iteration=8000 seq=50 bs =32  
***Average PCE: 0.5544908551655956  
Bracket: 0.018871589931451726 (680.0 / 36033.0)  
Change_edge: 0.017506904460983946 (374.0 / 21363.0)  
Chasse: 0.0 (0.0 / 5850.0)  
Choctaw: 0.015135600980312946 (562.0 / 37131.0)  
Counter_turn: 0.12507535102550907 (8507.0 / 68015.0)  
Cross_roll: 0.0002945508100147275 (3.0 / 10185.0)  
Loop: 0.07062106806678965 (4479.0 / 63423.0)  
Mohawk: 0.012738853503184714 (114.0 / 8949.0)  
Rocker_turn: 0.10760857851061537 (5991.0 / 55674.0)  
Three_turn: 0.00017254394478593768 (8.0 / 46365.0)  
Toe_step: 0.0013379947343433035 (31.0 / 23169.0)  
Twizzle: 0.11253525009011302 (10615.0 / 94326.0)  
No_element: 0.8632615694246196 (641677.0 / 743317.0)  
iteration=8000, bs = 8 seq=300 openposeの画像***






## 今やってること
### pallas
iteration=3000, bs=8, seq=300 
### glacus
iteration=8000, bs=4, seq=500 

