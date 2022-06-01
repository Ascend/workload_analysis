#!/usr/bin/python

import acl
from framework.util import check


class Device:
    """
    该类主要进行资源的准备，包括设备的初始化以及最终的回收
    同时，对算子执行提供stream资源。
    """

    def __init__(self, index):
        self.id = index
        ret = acl.rt.set_device(self.id)
        check(ret, "acl.rt.set_device (id:{})".format(self.id))
        self._context, ret = acl.rt.create_context(self.id)
        check(ret, "acl.rt.create_context (id:{})".format(self._context))
        self._stream, ret = acl.rt.create_stream()
        check(ret, "acl.rt.create_stream (id:{})".format(self._stream))
        self.ops = list()

    def stream(self):
        return self._stream

    def context(self):
        return self._context

    def register(self, op):
        self.ops.append(op)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ops:
            for op in self.ops:
                op.force_del()

        if self._stream:
            ret = acl.rt.destroy_stream(self._stream)
            check(ret, "acl.rt.destroy_stream ({})".format(self._stream))
            self._stream = None

        if self._context:
            ret = acl.rt.destroy_context(self._context)
            check(ret, "acl.rt.destroy_context ({})".format(self._context))
            self._context = None
        ret = acl.rt.reset_device(self.id)
        check(ret, "acl.rt.reset_device (id:{})".format(self.id))
