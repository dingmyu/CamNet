rm exp/drivable/res34_pretrain/model/events.out.tfevents*                               
rm exp/drivable/res34_pretrain/model/*.log
tensorboard --logdir="exp/drivable/res34_pretrain/model" --port 9825
