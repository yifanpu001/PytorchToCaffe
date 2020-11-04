import os


# widths = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# input_sizes = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,]

widths = [7.0, 8.0, 9.0,]
input_sizes = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,]


"""
python /home/pyf/codeforascend/PytorchToCaffe/pyf_experiments/exp04/resnet_d_w_reso.py --block BasicBlock --depth_config 2 2 2 2 --width 2 --input_size 224 224
"""
for width in widths:
    for input_size in input_sizes:
        cmd = f'python /home/pyf/codeforascend/PytorchToCaffe/pyf_experiments/exp05/mobilenetv2_w_reso.py \
            --width {width} \
            --input_size {input_size} {input_size}'
        os.system(cmd)
            
