from pathlib import Path

from ymir_exc.util import convert_ymir_to_coco

from .voc import CocoDetection, make_coco_transforms


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided ymir path {root} does not exist'

    data_info = convert_ymir_to_coco(cat_id_from_zero=True)

    datasets = []
    for split in image_set:
        img_folder = data_info[split]['img_dir']
        ann_file = data_info[split]['ann_file']
        dataset = CocoDetection(split, img_folder, ann_file,
            transforms=make_coco_transforms(split), return_masks=False)
        datasets.append(dataset)
    return datasets
