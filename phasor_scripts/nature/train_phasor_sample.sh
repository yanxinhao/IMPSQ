###
###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-16 13:26:20
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/train_grid_phasor copy.sh
 # @Date: 2022-05-16 13:26:20
 # @Description: 
### 
###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-16 13:26:18
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/train_grid_phasor.sh
 # @Date: 2022-05-16 10:51:54
 # @Description: 
### 
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-16 10:46:59
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/train_pe.sh
 # @Date: 2022-05-16 10:39:50
 # @Description: 
### 
# source activate idr
source activate ndface
# cd /root/workspace/NeuralRendering/PhasorImage
cd /root/workspace/NeuralRendering/ImpSq/
export CUDA_VISIBLE_DEVICES=7
python scripts/train_2d_image_phasor.py --cfg_file phasor_sample.yaml --dataset nature