import os
import logging
import socket
import traceback

logger = logging.getLogger(__name__)


def exec_cmd(cmd, log_flag=False, bg=False):
    # if log_flag: logger.info(cmd)
    if bg:
        os.system('{} &'.format(cmd))
    else:
        os.system(cmd)


def wait(msg):
    logger.info(msg)
    logger.info("Press Enter to continue...")
    # noinspection PyCompatibility
    raw_input()


def create_dir(dir_path, log_flag=True):
    cmd = 'mkdir -p {dir_path}'.format(dir_path=dir_path)
    exec_cmd(cmd, log_flag)


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


def launchTensorBoard(path, port, blocking=False):
    if blocking:
        import os
        os.system('tensorboard --logdir {} --port {}'.format(path, port))
        print('Started tensorboard at http://%s:%s' % (get_ip_address(), str(port)))
    else:
        import threading
        t = threading.Thread(target=launchTensorBoard, args=([path, port, True]))
        t.start()
