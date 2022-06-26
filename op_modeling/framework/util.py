import logging

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


def check(ret, msg):
    if ret != 0:
        logging.error(msg)
        raise Exception("{} failed. ({})".format(msg, ret))
    else:
        logging.debug(msg)
