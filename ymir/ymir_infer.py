"""
ymir infer task entry point
"""
import os.path as osp
import sys
from typing import List

import torch
from easydict import EasyDict as edict
from PIL import Image
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import (YmirStage, get_merged_config, get_weight_files,
                           get_ymir_process)

import datasets.transforms as T
from methods.coat_w_ram import coat_lite_mini, coat_lite_small, coat_lite_tiny
from methods.swin_w_ram import (swin_base_win7, swin_large_win7, swin_nano,
                                swin_small, swin_tiny)


def build_model(args):
    if args.backbone_name == 'swin_nano':
        backbone, hidden_dim = swin_nano(pretrained=None)
    elif args.backbone_name == 'swin_tiny':
        backbone, hidden_dim = swin_tiny(pretrained=None)
    elif args.backbone_name == 'swin_small':
        backbone, hidden_dim = swin_small(pretrained=None)
    elif args.backbone_name == 'swin_base_win7_22k':
        backbone, hidden_dim = swin_base_win7(pretrained=None)
    elif args.backbone_name == 'swin_large_win7_22k':
        backbone, hidden_dim = swin_large_win7(pretrained=None)
    elif args.backbone_name == 'coat_lite_tiny':
        backbone, hidden_dim = coat_lite_tiny(pretrained=None)
    elif args.backbone_name == 'coat_lite_mini':
        backbone, hidden_dim = coat_lite_mini(pretrained=None)
    elif args.backbone_name == 'coat_lite_small':
        backbone, hidden_dim = coat_lite_small(pretrained=None)
    else:
        raise ValueError(f'backbone {args.backbone_name} not supported')

    if args.method == 'vidt':
        return build_vidt_model(args, backbone)
    elif args.method == 'vidt_wo_neck':
        return build_vidt_wo_neck_model(args, backbone)
    else:
        available_methods = ['vidt_wo_neck', 'vidt']
        raise ValueError(f'method {args.method} is not in {available_methods}')

def build_vidt_model(args, backbone):
    from methods.vidt.deformable_transformer import \
        build_deforamble_transformer
    from methods.vidt.detector import Detector
    from methods.vidt.fpn_fusion import FPNFusionModule

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.reduced_dim,
                          cross_indices=args.cross_indices)

    cross_scale_fusion = None
    if args.cross_scale_fusion:
        cross_scale_fusion = FPNFusionModule(backbone.num_channels, fuse_dim=args.reduced_dim)

    deform_transformers = build_deforamble_transformer(args)

    model = Detector(
        backbone,
        deform_transformers,
        num_classes=args.num_classes,
        num_queries=args.det_token_num,
        # two essential techniques used in ViDT
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        # three additional techniques (optionally)
        cross_scale_fusion=cross_scale_fusion,
        iou_aware=args.iou_aware,
        token_label=args.token_label,
        # distil
        distil=False if args.distil_model is None else True,
    )

    return model

def build_vidt_wo_neck_model(args, backbone):
    from methods.vidt_wo_neck.detector import Detector

    backbone.finetune_det(method=args.method,
                          det_token_num=args.det_token_num,
                          pos_dim=args.pos_dim,
                          cross_indices=args.cross_indices)

    model = Detector(
        backbone,
        reduced_dim=args.reduced_dim,
        num_classes=args.num_classes,
    )

    return model

def get_postprocessor(args):
    if args.method == 'vidt':
        from methods.vidt.postprocessor import PostProcess as postprocess_vidt
        return postprocess_vidt(args.dataset_file)
    elif args.method == 'vidt_wo_neck':
        from methods.vidt_wo_neck.postprocessor import \
            PostProcess as postprocess_vidt_wo_neck
        return postprocess_vidt_wo_neck()
    else:
        available_methods = ['vidt_wo_neck', 'vidt']
        raise ValueError(f'method {args.method} is not in {available_methods}')

class YmirModel(object):
    def __init__(self, cfg: edict):
        """model for ymir infer task

        build model and load weight, inference

        Args:
            cfg: the ymir merged config, view get_merged_config()

        Raises:
            Exception: No weight files specified
        """
        self.cfg = cfg
        self.init_detector()

        # for multiple tasks, mining first, then infer
        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            infer_task_idx = 1
            task_num = 2
        else:
            infer_task_idx = 0
            task_num = 1

        self.task_idx = infer_task_idx
        self.task_num = task_num

        self.conf_threshold = self.cfg.param.conf_threshold
        self.class_names= self.cfg.param.class_names

    def init_detector(self):
        """
        build model and load weight
        """
        weight_files = get_weight_files(self.cfg, suffix=('.pth'))
        weight_files = [f for f in weight_files if osp.basename(f).startswith('checkpoint')]

        if len(weight_files) == 0:
            raise Exception('No weight files specified')

        latest_weight_file = max(weight_files, key=osp.getctime)

        # save "model", "optimizer", "lr_scheduler", "epoch" and "args" in weight_file
        load_data = torch.load(latest_weight_file, map_location='cpu')

        args = load_data['args']
        self.model = build_model(args)
        self.model.load_state_dict(load_data['model'], strict=False)
        self.device = torch.device(args.device)
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.RandomResize([512], max_size=800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.postprocess = get_postprocessor(args)

    def infer(self, img: Image) -> List[rw.Annotation]:
        sample, _ = self.preprocess(img, None)
        img_w, img_h = img.size
        orig_target_sizes = torch.tensor([[img_h, img_w]], device=self.device)
        outputs = self.model([sample.to(self.device)])
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        results = self.postprocess(outputs, orig_target_sizes)

        anns=[]
        # for batch
        for r in results:
            # for bbox
            for conf, cls, (xmin, ymin, xmax, ymax)  in zip(r['scores'], r['labels'], r['boxes']):
                if conf < self.conf_threshold:
                    continue

                ann = rw.Annotation(class_name=self.class_names[int(cls)], score=conf, box=rw.Box(
                    x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))

                anns.append(ann)
        return anns

def main() -> int:
    cfg = get_merged_config()
    model = YmirModel(cfg)
    task_idx = model.task_idx
    task_num = model.task_num
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.PREPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))

    N = dr.items_count(env.DatasetType.CANDIDATE)
    infer_result = {}

    idx = -1

    monitor_gap = max(1, N // 1000)
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        # img = cv2.imread(asset_path)
        img = Image.open(asset_path).convert('RGB')
        result = model.infer(img)
        infer_result[asset_path] = result
        idx += 1

        if idx % monitor_gap == 0:
            percent = get_ymir_process(
                stage=YmirStage.TASK, p=idx / N, task_idx=task_idx, task_num=task_num)
            monitor.write_monitor_logger(percent=percent)

    rw.write_infer_result(infer_result=infer_result)
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.PREPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))
    return 0

if __name__ == '__main__':
    sys.exit(main())
