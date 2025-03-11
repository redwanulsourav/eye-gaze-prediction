python generate_run_cfgs.py -s 32 -l 1 -t 0 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 1 -t 1 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 1 -t 2 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4

python generate_run_cfgs.py -s 32 -l 2 -t 0 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 2 -t 1 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 2 -t 2 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4

python generate_run_cfgs.py -s 32 -l 3 -t 0 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 3 -t 1 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 3 -t 2 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4

python generate_run_cfgs.py -s 32 -l 4 -t 0 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 4 -t 1 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 4 -t 2 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4

python generate_run_cfgs.py -s 32 -l 5 -t 0 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 5 -t 1 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4
python generate_run_cfgs.py -s 32 -l 5 -t 2 -d GTEA -v 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 -o 0 -r 0.0001 -b 6 -f -e 4

python generate_run_cfgs.py -s 3 -l $1 -t 0
python generate_run_cfgs.py -s 4 -l $1 -t 0
python generate_run_cfgs.py -s 5 -l $1 -t 0
python generate_run_cfgs.py -s 10 -l $1 -t 0
python generate_run_cfgs.py -s 15 -l $1 -t 0

python generate_run_cfgs.py -s 1 -l $1 -t 1
python generate_run_cfgs.py -s 2 -l $1 -t 1
python generate_run_cfgs.py -s 3 -l $1 -t 1
python generate_run_cfgs.py -s 4 -l $1 -t 1 
python generate_run_cfgs.py -s 5 -l $1 -t 1 
python generate_run_cfgs.py -s 10 -l $1 -t 1
python generate_run_cfgs.py -s 15 -l $1 -t 1


python generate_run_cfgs.py -s 1 -l $1 -t 2
python generate_run_cfgs.py -s 2 -l $1 -t 2
python generate_run_cfgs.py -s 3 -l $1 -t 2 
python generate_run_cfgs.py -s 4 -l $1 -t 2
python generate_run_cfgs.py -s 5 -l $1 -t 2 
python generate_run_cfgs.py -s 10 -l $1 -t 2 
python generate_run_cfgs.py -s 15 -l $1 -t 2




echo "python train.py -c run_cfgs/s=1,l=$1,t=0_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=2,l=$1,t=0_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=3,l=$1,t=0_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=4,l=$1,t=0_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=5,l=$1,t=0_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=10,l=$1,t=0_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=15,l=$1,t=0_o.yaml" >> script.sh


echo "python train.py -c run_cfgs/s=1,l=$1,t=1_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=2,l=$1,t=1_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=3,l=$1,t=1_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=4,l=$1,t=1_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=5,l=$1,t=1_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=10,l=$1,t=1_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=15,l=$1,t=1_o.yaml" >> script.sh


echo "python train.py -c run_cfgs/s=1,l=$1,t=2_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=2,l=$1,t=2_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=3,l=$1,t=2_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=4,l=$1,t=2_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=5,l=$1,t=2_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=10,l=$1,t=2_o.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=15,l=$1,t=2_o.yaml" >> script.sh

