##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash
# -- caveats --
# * make sure key-pair login is available

# -- config remote working dir
server='rackjesh'
remote_dir='/data3/chengyu/text_classification'
remote_data_dir='/data1/chengyu/bert_classification/data'
remote_env='bert2'
ssh $server "mkdir -p $remote_dir"
ssh $server "mkdir -p $remote_dir/checkpoints"
ssh $server "mkdir -p $remote_dir/logs"
ssh $server "mkdir -p $remote_dir/tmp"

# -- config job
config=$1 
[[ -z $config ]] && config="config.yaml"
sed -i "s@^data_dir:.*@data_dir: '$remote_data_dir'@g" $config # use @ as deliminator because slash / conflicts with path
python config.py -c $config --remote --server $server --remote_dir $remote_dir
[[ $? -ne 0 ]] && echo 'exit' && exit 2
checkpoint=$(cat tmp/path.tmp)
path="$remote_dir/checkpoints/$checkpoint"
echo $path
# TODO: directly upload json in config.py
scp "tmp/para.json" $server:$path

# -- copy src code
scp $config $server:$path"/config.yaml"
scp main.py $server:$path
scp config.py $server:$path
scp -r src $server:$path

# -- write run file
cat <<EOT > tmp/run.sh
trim() {
  local s2 s="\$*"
  until s2="\${s#[[:space:]]}"; [ "\$s2" = "\$s" ]; do s="\$s2"; done
  until s2="\${s%[[:space:]]}"; [ "\$s2" = "\$s" ]; do s="\$s2"; done
  echo "\$s"
}

source /home/chengyu/anaconda2/bin/activate $remote_env
cd $path
if [ \$(cat config.yaml | grep "^test:" | awk '{print\$2}') == 'True' ]; then
    python main.py | tee train.out
else
    python main.py > train.out 2>&1 &
fi
pid=\$!
gpu_id=\$(trim \$(grep 'gpu_id' config.yaml | awk -F ':' '{print\$2}'))
echo "[$server] [\$pid] [\$gpu_id] [Path]: $path"
echo "s [\$pid] [\$gpu_id] \$(date) [Path]: $path" >> $remote_dir/logs/log.txt
EOT

# -- remote run
scp tmp/run.sh $server:$path
ssh $server "cd $path && bash run.sh" >> logs/log.txt
cat logs/log.txt | tail -n1
