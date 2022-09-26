###
###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-18 13:57:56
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/text/train_phasor.sh
 # @Date: 2022-05-18 13:57:55
 # @Description: 
### 
###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2022-05-17 13:07:51
 # @LastEditors: yanxinhao
 # @FilePath: /ImpSq/phasor_scripts/train_phasor.sh
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
export CUDA_VISIBLE_DEVICES=1
python scripts/train_2d_image_fabric.py --cfg_file phasor.yaml --dataset fabrics