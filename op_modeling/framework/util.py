

def check(ret, msg):
    # print(msg)
    if ret != 0:
        raise Exception("{} failed. ({})".format(msg, ret))