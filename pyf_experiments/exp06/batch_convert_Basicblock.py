import os


blocks = ['BasicBlock']
depth_configs = [
    [2, 2, 2, 2],
    [3, 4, 6, 3],
]
widths = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
input_sizes = [50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700]

# depth_configs = [
#     [2, 2, 2, 2],
# ]
# widths = [0.75,]
# input_sizes = [125,]

"""
python /home/pyf/codeforascend/PytorchToCaffe/pyf_experiments/exp04/resnet_d_w_reso.py --block BasicBlock --depth_config 2 2 2 2 --width 2 --input_size 224 224
"""

for d in depth_configs:
    for width in widths:
        for input_size in input_sizes:
            cmd = f'python /home/pyf/codeforascend/PytorchToCaffe/pyf_experiments/exp06/resnet_d_w_reso.py \
                --block BasicBlock \
                --depth_config {d[0]} {d[1]} {d[2]} {d[3]} \
                --width {width} \
                --input_size {input_size} {input_size}'
            os.system(cmd)
            
