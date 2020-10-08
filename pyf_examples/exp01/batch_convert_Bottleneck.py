import os


blocks = ['Bottleneck']
depth_configs = [
    [2, 2, 2, 2],
    [3, 4, 6, 3],
    [3, 4, 23, 3],
    [3, 8, 36, 3],
]
widths = [1, 2, 3, 4]
input_sizes = [256, 512, 1024, 1536]

"""
python /home/pyf/codeforascend/PytorchToCaffe/pyf_examples/exp01/resnet_d_w_reso.py --block BasicBlock --depth_config 2 2 2 2 --width 2 --input_size 224 224
"""
for d in depth_configs:
    for width in widths:
        for input_size in input_sizes:
            cmd = f'python /home/pyf/codeforascend/PytorchToCaffe/pyf_examples/exp01/resnet_d_w_reso.py \
                --block BasicBlock \
                --depth_config {d[0]} {d[1]} {d[2]} {d[3]} \
                --width {width} \
                --input_size {input_size} {input_size}'
            os.system(cmd)
            
