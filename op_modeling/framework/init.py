#!/usr/bin/python

import atexit

import acl
from framework.util import check


def init():
    ret = acl.init()
    check(ret, "acl_init")

    def deinit():
        ret = acl.finalize()
        check(ret, "acl.finalize")

    atexit.register(deinit)



