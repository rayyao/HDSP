
##Installation
Create and activate a conda environment:
conda create -n hdsp python=3.7
conda activate hdsp
Install the required packages
bash install_hdsp.sh

##Path Setting
Run the following command to set paths
cd <PATH_of_HDSP>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output

##Training
Download the pretrained foundation model (OSTrack) and put it under ./pretrained/.
python tracking/train.py --script HDSP --config hdsp_deep --save_dir ./output --mode multiple --nproc_per_node 2

##Testing
Modify the <DATASET_PATH> and <SAVE_PATH>
python HSI_test/tsetHSI.py  --script_name HDSP --dataset_name HSItest --yaml_name hdsp_deep