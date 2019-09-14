
for((i=0;i<130;i++));  
do   
nohup srun -p AD -c2  python -u camnet_get_triple.py $(expr $i \* 200) $(expr $i \* 200 + 200)  2>&1 > triple_lists/$i.list & 
done
