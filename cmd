#A100
ssh -p 22 ruilan@36.212.209.151

#copy result
scp -P 22 ruilan@36.212.209.151:/data/wanglong/github/SpatialLM/scene0000_00.rrd ./

conda activate spatillm
rerun scene0000_00.rrd

# generate normalized scenescript
python tools/layout_parser.py data/layout/scenes-04_00-06_00 
python tools/layout_parser.py data/layout/all-scenes.txt 0 10 >data/layout/scenes-normalized_0-10.txt

#prompt for openai model to generate description
帮我处理一批房间布局，仅给出布局说明即可，采用json输出，key为场景名，value是一个dict，又包含一个 k为 description，v是布局说明，脚本如下:


#merge script and description
python tools/merge_script_and_desc.py ./data/layout/normalized 

#deps
#pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

#create dataset
python spatiallm/tuner/create_llm_dataset.py \
  --dataset_dir arkitscenes-llm \
  --split_csv arkitscenes-llm/split.csv \
  --dataset_name arkitscenes \
  --code_template_file code_template.txt
#inference (bbox detection)
python inference_val.py 
#visualize detection results
python visualize.py \
  --point_cloud /data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/pcd/scene_002.json.ply \
  --layout /data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/layout/scene_002.txt \
  --serve

#finetune model
export WANDB_API_KEY=fe32d193c91d04565a7918c276d5c0acebb78b05
export CUDA_VISIBLE_DEVICES=0,1
rm -rf saves/arkitscenes
tmux new -s train_model
tmux new -s llama
tmux new -s llama2
conda activate spatiallm
llamafactory-cli train deepseek_full_sft.yaml
DS_SKIP_CUDA_CHECK=1 llamafactory-cli train deepseek_full_loss_sft.yaml
tmux attach -t train_model
tmux attach -t llama2



pip uninstall spatiallm
pip install -e . --no-build-isolation
pip list | grep spatiallm