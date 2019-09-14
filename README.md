# CamNet
A pytorch implementation of "CamNet: Coarse-to-Fine Retrieval for Camera Re-Localization, ICCV 2019"

## CamNet: Coarse-to-Fine Retrieval for Camera Re-Localization

### DataPrepare
Download the dataset and extract into folder scripts/

cd scripts

sh camnet_get_triple.sh

cat triple_lists/\*.list | grep seq > triple.list

mv triple.list into Train/data_list/ folder

### PreTrain
cd preTrain

sh tensorboard.sh

source source.sh

sh train.sh

mv the pretrain model into Train/ folder

### Train
cd Train/

source source.sh

sh train.sh

### Evaluation
source source.sh

sh eval.sh

cd filter_pose

sh run.sh
