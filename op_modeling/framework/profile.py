#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging
import os
import stat
import shutil
import random

import csv
import glob
import pandas as pd

import acl
from termcolor import cprint

from framework.ai_metric import AICoreMetric
from framework.util import check


def create_tmp_dir():
    """
    需要随机生成文件夹，并在结束时删除
    :return:
    """
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tmp_dir = os.path.join(output_path, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    filename = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba', 16))
    tmp_dir = os.path.join(tmp_dir, filename)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir


class Profile:
    """
    该类主要提供算子的profiling能力，最终将数据汇聚到统一的文件中

    具体实现：
    由于ACL的profiling本身的限制，其会在特定目录下生成自定义的文件夹，流程需要保证生成新的文件夹后，需要立刻进行解析，并生成结果文件。
    """

    def __init__(self, device, dump_file, op_type, interval=25):
        self.tmp_dir = create_tmp_dir()
        self.dump_file = dump_file
        ret = acl.prof.init(self.tmp_dir)
        check(ret, "acl.prof.init")

        self.device = device
        device_list = [device.id]
        # 最后一个参数标识收集内容，最大为1FF, 此处6是现在收集的内容最小集
        self.config = acl.prof.create_config(device_list, AICoreMetric.PIPE_UTILIZATION.value, 0, 6)
        self.current_origin_op_info = list()
        self.interval = interval

        # dump file
        self.op_type = op_type
        if not os.path.exists(os.path.dirname(dump_file)):
            raise Exception(f"please create {dump_file}")
        self.file_handle = os.fdopen(os.open(dump_file, os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), 'w')
        self.prof_info = ['Input Shapes', 'Input Data Types', 'Input Formats',
                          'Output Shapes', 'Output Data Types', 'Output Formats',
                          'Block Dim', 'aicore_time(us)',
                          'mac_ratio', 'vec_ratio', 'scalar_ratio',
                          'mte1_ratio', 'mte2_ratio', 'mte3_ratio']
        self.origin_info = ['Original Inputs', 'Original Outputs', "Attributes"]
        self.dict_writer = csv.DictWriter(self.file_handle, fieldnames=self.origin_info + self.prof_info)
        self.dict_writer.writeheader()
        self._start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
        if self.count:
            self._dump()

        ret = acl.prof.destroy_config(self.config)
        check(ret, "acl.prof.destroy_config")
        ret = acl.prof.finalize()
        check(ret, "acl.prof.finalize")
        self.file_handle.close()
        shutil.rmtree(self.tmp_dir)

    def add_op_origin_desc(self, desc):
        self.current_origin_op_info.append(desc)

    def set_interval(self, interval):
        self.interval = interval

    def inc(self):
        self.count += 1
        if self.count % self.interval:
            return

        self._stop()
        self._dump()
        self._start()

    def _start(self):
        # 移除旧的profiling数据
        ret = acl.prof.start(self.config)
        check(ret, "acl.prof.start")
        self.count = 0

    def _stop(self):
        ret = acl.prof.stop(self.config)
        check(ret, "acl.prof.stop")

    def _parse_profiling_data(self):
        def data_generate(data_path):
            tmp_paths = os.listdir(f"{data_path}")
            tmp_path = os.path.join(data_path, tmp_paths[0])  # 由于本身仅有1个文件夹在该目录中，故而此处特殊处理
            os.system(f"msprof --export=on --output={tmp_path} > /dev/null")
            return tmp_path

        def data_parser(prof_data_path, op_type):
            op_summary_file = glob.glob(f"{prof_data_path}/device_{self.device.id}/summary/op_summary_*.csv")[0]
            df = pd.read_csv(op_summary_file)
            # 从profiling数据中筛选目标type的算子，避免Cast，TransData等算子的干扰
            df = df[df["OP Type"] == op_type]
            tab = df[self.prof_info]
            prof_infos = list()
            for i in range(tab.shape[0]):
                prof_data = dict(tab.iloc[i, :])
                prof_infos.append(prof_data)
            return prof_infos

        def combine_origin_and_prof(origin_op_info, prof_infos):
            combined_info = []
            for op_desc, prof_info in zip(origin_op_info, prof_infos):
                combined_info.append({**op_desc, **prof_info})
            return combined_info

        prof_data_path = data_generate(self.tmp_dir)
        prof_infos = data_parser(prof_data_path, self.op_type)
        if len(self.current_origin_op_info) == 0:
            return prof_infos

        # 部分算子未执行或其他原因
        if len(prof_infos) != len(self.current_origin_op_info):
            raise Exception(f"Size mismatch for ops({len(self.current_origin_op_info)}) "
                            f"and profiling results({len(prof_infos)})")

        prof_infos = combine_origin_and_prof(self.current_origin_op_info, prof_infos)
        return prof_infos

    def _dump(self):
        cur_prof_infos = []
        try:
            cur_prof_infos = self._parse_profiling_data()
        except Exception as e:
            logging.error(f"dump failed: {e}")
            self._save_err_msg(self.current_origin_op_info)

        for prof_info in cur_prof_infos:
            self.dict_writer.writerow(prof_info)
        self.file_handle.flush()
        self.current_origin_op_info.clear()
        # 解析后，临时文件夹中内容已经不需要
        shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)

    def _save_err_msg(self, op_desc):
        class ErrorMsg:
            """
            该类用于保存profiling出错时的信息，包括
            1. 算子原始的描述
            2. profiling原始数据
            """

            def __init__(self, profile_dir, dump_dir, err_op_desc):
                """
                :param profile_dir: profile原始数据所在文件夹 (*/PROF_XX)
                :param dump_dir: 错误数据保存的目标文件夹
                :param err_op_desc: 算子原始的描述
                """
                self.profile_dir = profile_dir
                self.dump_dir = dump_dir
                profile_name = os.path.basename(self.profile_dir)
                self.profile_save_path = os.path.join(self.dump_dir, profile_name)
                self.err_origin_info_file = os.path.join(self.dump_dir, profile_name, "err_origin_info.csv")
                self.err_op_desc = err_op_desc

            def save(self):
                try:
                    shutil.copytree(self.profile_dir, self.profile_save_path)
                except Exception as e:
                    logging.error(f"Save original profile data error: {e}")
                    return False
                self.err_op_desc.to_csv(self.err_origin_info_file, index=True)
                return True

        prof_dir = glob.glob(f"{self.tmp_dir}/PROF*")[0]
        error_save_dir = os.path.join(os.path.dirname(self.dump_file), 'error_prof')
        error_msg = ErrorMsg(prof_dir, error_save_dir, op_desc)
        status = error_msg.save()
        if status:
            cprint(f"Save error msg {prof_dir} in {error_save_dir}", on_color="on_green")
