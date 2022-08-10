import logging
import sys
from easydict import EasyDict as edict
import os
from ymir_exc.util import get_merged_config
import os.path as osp
import subprocess

def main(cfg: edict) -> int:
    num_gpus = len(cfg.param.gpu_id.split(','))
    models_dir = cfg.ymir.output.models_dir
    args_options = cfg.param.get('args_options','')
    num_classes = len(cfg.param.class_names)
    batch_size = int(cfg.param.batch_size)
    eval_size = int(cfg.param.eval_size)
    epochs = int(cfg.param.epochs)
    num_workers = int(cfg.param.num_workers)
    learning_rate = float(cfg.param.learning_rate)
    backbone_name = str(cfg.param.backbone_name)

    cmd = f"""
    python -m torch.distributed.launch \
       --nproc_per_node {num_gpus} \
       --nnodes 1 \
       --use_env main.py \
       --method vidt \
       --backbone_name {backbone_name} \
       --epochs {epochs} \
       --lr {learning_rate} \
       --min-lr 1e-7 \
       --batch_size {batch_size} \
       --num_workers {num_workers} \
       --eval_size {eval_size} \
       --aux_loss True \
       --with_box_refine True \
       --dataset_file ymir \
       --num_classes {num_classes} \
       --coco_path /in \
       --output_dir {models_dir}
       """

    if args_options:
        cmd = cmd + " " + args_options

    logging.info(f"Running command: {cmd}")
    subprocess.run(cmd.split(), check=True)

    write_ymir_training_result(last=True)

    return 0

if __name__ == '__main__':
    cfg = get_merged_config()
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    os.environ.setdefault('EVAL_TMP_FILE', osp.join(cfg.ymir.output.models_dir, 'eval_tmp.json'))
    os.environ.setdefault('YMIR_MODELS_DIR', cfg.ymir.output.models_dir)
    os.environ.setdefault('TENSORBOARD_DIR', cfg.ymir.output.tensorboard_dir)
    sys.exit(main(cfg))
