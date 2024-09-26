import logging
import subprocess
import sys

from ymir_exc import monitor
from easydict import EasyDict as edict
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process


def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training()
    elif cfg.ymir.run_mining or cfg.ymir.run_infer:
        # for multiple tasks, run mining first, infer later.

        if cfg.ymir.run_mining:
            _run_mining(cfg)
        if cfg.ymir.run_infer:
            _run_infer(cfg)
    else:
        logging.warning('no task running')

    return 0


def _run_training() -> None:
    command = 'python3 ymir/ymir_training.py'
    logging.info(f'training: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict) -> None:
    command = 'python3 ymir/ymir_mining.py'
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    write_ymir_monitor_process(cfg, task='mining', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)


def _run_infer(cfg: edict) -> None:
    command = 'python3 ymir/ymir_infer.py'
    logging.info(f'infer: {command}')
    subprocess.run(command.split(), check=True)
    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    sys.exit(start())
