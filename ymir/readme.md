# ymir vidt documence

## requirements
```
onnxruntime 1.12.1 has requirement numpy>=1.21.0
```

## ymir supports

### ymir dataset supports
add new option dataset_file=ymir and num_classes for args, modify related files.

### ymir training supports

- ymir/ymir_training.py, entry point for ymir training task,
- ymir/img-man/training-template.py hyper-param template
```
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
```

- save weight and map50, monitor process write tensorboard logs

```
from tensorboardX import SummaryWriter
from ymir_exc import monitor
from ymir_exc.util import (YmirStage, get_merged_config, get_ymir_process,
                           write_ymir_training_result)

# save weight, map and monitor process for main process.
tb_writer = SummaryWriter(args.tensorboard_dir)
for epoch in range(args.start_epoch, args.epochs):
    # monitor process
    percent = get_ymir_process(stage=YmirStage.TASK, p=(epoch - args.start_epoch + 1) / (args.epochs - args.start_epoch + 1))
    monitor.write_monitor_logger(percent=percent)

    # save weight and map
    map50 = test_stats['coco_eval_bbox'][1]
    write_ymir_training_result(cfg, map50, files=checkpoint_paths, id=str(epoch))

    # writer tensorboard logs
    tb_writer_add_scaler(tag='train/loss', scalar_value=0.1, global_step=epoch)
```

- finetune

```
# finetune from offered weight file
weight_files = get_weight_files(cfg, suffix=('.pth'))
if weight_files:
    latest_weight_file = max(weight_files, key=osp.getctime)

    # auto finetune if not designed by user.
    if args_options.find('--load_from')== -1 and args_options.find('--resume')==-1:
        cmd = cmd + " --load_from " + latest_weight_file
logging.info(f"Running command: {cmd}")
subprocess.run(cmd.split(), check=True)

if args.load_from:
    if args.resume:
        raise Exception("cannot load from and resume at the same time")

    if args.load_from.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.load_from, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.load_from, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print('load a checkpoint from', args.load_from)
```

## thanks to

- [detr](https://github.com/facebookresearch/detr)
- [vidt](https://github.com/naver-ai/vidt)
