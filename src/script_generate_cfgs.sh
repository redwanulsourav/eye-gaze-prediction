python generate_run_cfgs.py -s 1 -l $1 -t 0
python generate_run_cfgs.py -s 2 -l $1 -t 0
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



echo "python train.py -c run_cfgs/s=1,l=$1,t=0.yaml" > script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=0.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=2,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=0.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=3,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=0.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=4,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=0.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=5,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=0.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=10,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=0.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=15,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=0.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=20,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=0.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=0.yaml" >> script.sh


echo "python train.py -c run_cfgs/s=1,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=2,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=3,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=4,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=5,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=10,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=15,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=20,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=1.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=1.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=1,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=1,l=$1,t=2.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=2,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=2,l=$1,t=2.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=3,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=3,l=$1,t=2.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=4,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=4,l=$1,t=2.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=5,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=5,l=$1,t=2.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=10,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=10,l=$1,t=2.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=15,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=15,l=$1,t=2.yaml" >> script.sh

echo "python train.py -c run_cfgs/s=20,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=2.yaml" >> script.sh
echo "python train.py -c run_cfgs/s=20,l=$1,t=2.yaml" >> script.sh

