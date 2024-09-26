import logging
import os
import os.path as osp
import subprocess
import sys

from easydict import EasyDict as edict
from ymir_exc.util import get_merged_config, get_weight_files, write_ymir_training_result


def main(cfg: edict) -> int:
    num_gpus = len(cfg.param.gpu_id.split(','))
    models_dir = cfg.ymir.output.models_dir
    tensorboard_dir = cfg.ymir.output.tensorboard_dir
    args_options = cfg.param.get('args_options', '')
    num_classes = len(cfg.param.class_names)
    # gpu_count = len(cfg.param.get('gpu_id', '0').split(','))
    batch_size = int(cfg.param.batch_size_per_gpu)
    eval_size = int(cfg.param.eval_size)
    epochs = int(cfg.param.epochs)
    num_workers = int(cfg.param.num_workers_per_gpu)
    learning_rate = float(cfg.param.learning_rate)
    backbone_name = str(cfg.param.backbone_name)
    weight_save_interval = int(cfg.param.weight_save_interval)

    cmd = f"""
    python -m torch.distributed.launch \
       --nproc_per_node {num_gpus} \
       --nnodes 1 \
       --use_env main.py \
       --backbone_name {backbone_name} \
       --epochs {epochs} \
       --lr {learning_rate} \
       --batch_size {batch_size} \
       --num_workers {num_workers} \
       --eval_size {eval_size} \
       --dataset_file ymir \
       --num_classes {num_classes} \
       --coco_path /in \
       --output_dir {models_dir} \
       --save_interval {weight_save_interval} \
       --tensorboard_dir {tensorboard_dir}
       """

    if args_options:
        cmd = cmd + " " + args_options

    # finetune from offered weight file
    weight_files = get_weight_files(cfg, suffix=('.pth'))
    weight_files = [
        f for f in weight_files if osp.basename(f).startswith('checkpoint')
    ]
    if weight_files:
        latest_weight_file = max(weight_files, key=osp.getctime)

        # auto finetune if not specified by user.
        if args_options.find('--load_from') == -1 and args_options.find(
                '--resume') == -1:
            cmd = cmd + " --load_from " + latest_weight_file

    logging.info(f"Running command: {cmd}")
    subprocess.run(cmd.split(), check=True)

    write_ymir_training_result(cfg, map50=0, files=[], id='last')

    return 0


if __name__ == '__main__':
    cfg = get_merged_config()
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(main(cfg))
