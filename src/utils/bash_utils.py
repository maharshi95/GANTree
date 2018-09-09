import os
import logging

logger = logging.getLogger(__name__)


def exec_cmd(cmd, log=True):
    if log: logger.info(cmd)
    os.system(cmd)


def wait(msg):
    logger.info(msg)
    logger.info("Press Enter to continue...")
    raw_input()


def create_dir(dir_path):
    cmd = 'mkdir -p {dir_path}'.format(dir_path=dir_path)
    exec_cmd(cmd)


def clear_dir(dir_path):
    cmd = 'rm -r {dir_path}/*'.format(dir_path=dir_path)
    exec_cmd(cmd)


def delete_file(filepath):
    cmd = 'rm %s' % filepath
    exec_cmd(cmd)


def delete_recursive(dir_path, force=False):
    cmd = 'rm -rf {}' if force else 'rm -r {}'
    exec_cmd(cmd.format(dir_path))


def copy_file(src_file, destination):
    cmd = 'cp %s %s' % (src_file, destination)
    exec_cmd(cmd)


def copy_files(files, destination):
    for f in files:
        copy_file(f, destination)
