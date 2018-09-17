import os
import logging
import socket
import traceback
from contextlib import closing

import paths

logger = logging.getLogger(__name__)


def exec_cmd(cmd, log=True, bg=False):
    if log: logger.info(cmd)
    if bg:
        os.system('{} &'.format(cmd))
    else:
        os.system(cmd)


def wait(msg):
    logger.info(msg)
    logger.info("Press Enter to continue...")
    # noinspection PyCompatibility
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


def find_free_port(base_port=7000):
    port = base_port
    available = is_port_available(port)
    while not available:
        port += 1
        available = is_port_available(port)
    return port


def is_port_available(port=7000):
    import socket, errno

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind(("0.0.0.0", port))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            return False
        else:
            traceback.print_exc()
            raise Exception(e)
    s.close()
    return True


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def start_tensorboard(base_port):
    available_port = find_free_port(base_port)
    cmd = "tensorboard --logdir {} --port {}".format(paths.logs_base_dir, available_port)
    ip = get_ip_address()
    logger.info("Starting Tensorboard for experiment {} at http://{}:{}".format(paths.exp_name, ip, available_port))
    # exec_cmd(cmd, bg=True)
    return ip, available_port
