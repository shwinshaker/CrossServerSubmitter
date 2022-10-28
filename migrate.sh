##################################################
# File  Name: migrate.sh
#     Author: shwin
# Creat Time: Thu 26 Nov 2020 10:59:03 AM PST
##################################################

#!/bin/bash

paths=( 
        '/data3/zichao/backdoor2/data/imdb/IMDB.csv'
)

new_paths=(
        'data/imdb/IMDB.csv'
)

for ((i=0; i<${#paths[@]}; i++));do
    echo
    echo ">>>>>>>> "${new_paths[$i]}
    if [ -f "./${new_paths[$i]}" ] || [ -d "./${new_paths[$i]}" ]; then
        read -p "path ${new_paths[$i]} already exists. Delete[d], Continue[c], Skip[s] or Terminate[*]? " ans
        case $ans in
           d ) rm -rf ${new_paths[$i]};;
           c ) ;;
           s ) continue;;
           * ) exit;;
        esac
        # exit -1
    fi
    # scp -r chengyu@rackjesh:/home/chengyu/bert_classification/${paths[$i]} ./${new_paths[$i]}
    # scp -r chengyu@fifth:/home/chengyu/bert_classification/${paths[$i]} ./${new_paths[$i]}
    scp -r chengyu@rackjesh:${paths[$i]} ${new_paths[$i]}
    # scp -r chengyu@fifth:/home/chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # scp -r chengyu@descartes:/home/chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # scp -r jingbo@deepx:/home/jingbo/Chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # rsync -avP --exclude='*.pth.tar' --exclude='ad_running_avg.npy' ftiasch@wu02-1080ti:/home/ftiasch/chengyu/Initialization/${paths[$i]}/ ./${new_paths[$i]}
    # rsync -avP --exclude='*.pth.tar' --exclude='ad_running_avg.npy' chengyu@rackjesh:/home/chengyu/Initialization/${paths[$i]}/ ./${new_paths[$i]}
    # rsync -avP ftiasch@wu02-1080ti:/home/ftiasch/chengyu/Initialization/${paths[$i]}/ ./${new_paths[$i]}
done
