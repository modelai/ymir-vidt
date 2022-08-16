import logging
import subprocess
import sys

from ymir_exc import monitor
from ymir_exc.util import YmirStage, get_merged_config, get_ymir_process


def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training()
    elif cfg.ymir.run_mining or cfg.ymir.run_infer:
        # for multiple tasks, run mining first, infer later.
        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            task_num = 2
            mining_task_idx = 0
            infer_task_idx = 1
        else:
            task_num = 1
            mining_task_idx = 0
            infer_task_idx = 0

        if cfg.ymir.run_mining:
            _run_mining(task_idx=mining_task_idx, task_num=task_num)
        if cfg.ymir.run_infer:
            _run_infer(task_idx=infer_task_idx, task_num=task_num)
    else:
        logging.warning('no task running')

    return 0


def _run_training() -> None:
    command = 'python3 ymir/ymir_training.py'
    logging.info(f'training: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(task_idx: int = 0, task_num: int = 1) -> None:
    command = f'python3 ymir/ymir_mining.py'
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.POSTPROCESS, p=1, task_idx=task_idx,
        task_num=task_num))


def _run_infer(task_idx: int = 0, task_num: int = 1) -> None:
    command = f'python3 ymir/ymir_infer.py'
    logging.info(f'infer: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.POSTPROCESS, p=1, task_idx=task_idx,
        task_num=task_num))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    sys.exit(start())
