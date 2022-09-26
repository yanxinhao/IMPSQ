###
###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-18 17:13:38
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/text/train_siren.sh
 # @Date: 2022-05-18 17:11:17
 # @Description: 
### 
###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-18 17:10:40
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/train_pe copy.sh
 # @Date: 2022-05-18 17:10:40
 # @Description: 
### 
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-18 14:01:30
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/train_pe.sh
 # @Date: 2022-05-16 10:39:50
 # @Description: 
### 
# source activate idr
source activate ndface
# cd /root/workspace/NeuralRendering/PhasorImage
cd /root/workspace/NeuralRendering/ImpSq/
export CUDA_VISIBLE_DEVICES=4
python scripts/train_2d_image_phasor.py --cfg_file siren.yaml --dataset nature