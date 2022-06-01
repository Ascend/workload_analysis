#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import traceback

import acl
import glob
import os
import pandas as pd
import shutil
import random

from termcolor import cprint

from framework.ai_metric import AICoreMetric
from framework.data_manager import DataManager
from framework.util import check, exec_subprocess


def create_tmp_dir(device_id, output_path):
    """
    需要随机生成文件夹，并在结束时删除
    :return:
    """
    # output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tmp_dir = os.path.join(output_path, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    t = 1000 * time.time()
    random.seed(int(t) % 2 ** 32)
    filename = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba', 16)) + str(device_id)
    tmp_dir = os.path.join(tmp_dir, filename)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir


def filter_by_col(df: pd.DataFrame, col_name, value):
    valid_index = []
    k = 0
    for i, row in df.iterrows():
        if k >= len(value):
            break

        if row[col_name] == value[k]:
            k = k + 1
            valid_index.append(i)

    ret = df.iloc[valid_index, :]
    return ret


class Profile:
    """
    该类主要提供算子的profiling能力，最终将数据汇聚到统一的文件中

    具体实现：
    由于ACL的profiling本身的限制，其会在特定目录下生成自定义的文件夹，流程需要保证生成新的文件夹后，需要立刻进行解析，并生成结果文件。
    """

    def __init__(self, device, data_manager: DataManager, ai_metric=AICoreMetric.PIPE_UTILIZATION,
                 output_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"),
                 interval=25):
        self.data_manager = data_manager
        self.ai_metric = ai_metric
        self.interval = interval
        self.tmp_dir = create_tmp_dir(device.id, output_path)
        self.error_prof_dir = os.path.join(os.path.dirname(self.data_manager.dump_file), 'error_prof')
        if not os.path.exists(self.error_prof_dir):
            os.makedirs(self.error_prof_dir)
        ret = acl.prof.init(self.tmp_dir)
        check(ret, "acl.prof.init")
        self.device = device
        device_list = [device.id]
        # 310 不支持l2
        self.config = acl.prof.create_config(device_list, ai_metric.value, 0, 0x16)  # 最后一位标识收集内容，最大为4F, 此处6是现在收集的内容最小集
        # self.prof_info = ['Input Shapes', 'Input Data Types', 'Input Formats', "Block Dim", 'aicore_time(us)']
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
        shutil.rmtree(self.tmp_dir)

    def _start(self):
        # 移除旧的profiling数据
        ret = acl.prof.start(self.config)
        check(ret, "acl.prof.start")
        self.count = 0

    def _stop(self):
        ret = acl.prof.stop(self.config)
        check(ret, "acl.prof.stop")

    def check_sample_exists(self, op_type, origi_inputs, origi_outputs, attributes, op_name=None):
        return self.data_manager.check_sample_exists(op_type, origi_inputs, origi_outputs, attributes, op_name=op_name)

    def _parse_profiling_data(self, op_names, op_types):
        def data_generate(data_path):
            tmp_paths = os.listdir(f"{data_path}")

            tmp_path = os.path.join(data_path, tmp_paths[0])  # 由于本身仅有1个文件夹在该目录中，故而此处特殊处理
            os.system(f"msprof --export=on --output={tmp_path} > /dev/null")
            os.system("sleep 2")  # 保证数据保存下来

            return tmp_path

        def data_parser(prof_data_path):
            # 获取目录列表
            if os.path.exists(f"{prof_data_path}/device_{self.device.id}"):
                # 新版本CANN路径
                summary_path = f"{prof_data_path}/device_{self.device.id}/summary"
            else:
                # 老版本CANN路径
                summary_path = f"{prof_data_path}/summary"

            op_summary_files = glob.glob(f"{summary_path}/op_summary_*.csv")
            l2_cache_files = glob.glob(f"{summary_path}/l2_cache_*.csv")

            if len(op_summary_files) == 0:
                raise Exception(f"No op summary file found in {summary_path}")

            op_summary_file = op_summary_files[0]

            # 处理AI Core相关信息
            ai_core_df = pd.read_csv(op_summary_file)
            # 只需要类型为self.op_type的Profiling结果，过滤掉trans_data算子等
            # 单算子场景中prof的'Op Name'和'OP Type'列一般来说相同（除非出现算子转化等特殊场景，此时Op Name不变，OP Type变化）
            ai_core_df = filter_by_col(ai_core_df, 'OP Type', op_types)

            tab = ai_core_df

            if ai_core_df.empty:
                print(
                    "[Warning] the profiling result for op type: {} is empty!".format(str(op_types)))

            # 处理l2_cache文件, 只有910和710有l2_cache, 将l2_cache信息合并到结果
            if len(l2_cache_files) > 0:
                l2_cache_df = pd.read_csv(l2_cache_files[0])
                l2_cache_df = filter_by_col(l2_cache_df, 'Op Name', op_names)

                if len(ai_core_df) != len(l2_cache_df):
                    cprint("Mismatch size for aicore data: {} and l2 cache data: {},"
                           " l2 cache for this prof collection will be Discarded".
                           format(len(ai_core_df), len(l2_cache_df)), on_color='on_red')
                    return tab
                else:
                    tab = pd.concat((tab, l2_cache_df[["Hit Rate", "Victim Rate"]]), axis=1)

            return tab

        prof_data_path = data_generate(self.tmp_dir)
        prof_df = data_parser(prof_data_path)
        return prof_df

    def _dump(self):
        try:
            prof_df = self._parse_profiling_data(self.data_manager.op_names, self.data_manager.op_types)
            self.data_manager.append(prof_df)
            self.data_manager.dump()

            # 解析后，临时文件夹中内容已经不需要
            shutil.rmtree(self.tmp_dir)
            os.mkdir(self.tmp_dir)
        except Exception as e:
            traceback.print_exc()

            cprint(f"Dump file Error: {e}", on_color="on_red")
            # traceback.print_exc()
            if os.path.exists(self.tmp_dir):
                # 保存出错的profiling原始数据
                prof_d = glob.glob(f"{self.tmp_dir}/PROF*")[0]
                dir_name = os.path.basename(prof_d)
                desc_d = f"{self.error_prof_dir}/{dir_name}"
                shutil.move(prof_d, desc_d)

                # 将可能存在的其他错误信息与profiling数据一起保存
                self.data_manager.err_msg.set_dump_dir(desc_d)
                self.data_manager.err_msg.save()
                cprint(f"Save exception data {prof_d} in {desc_d}", on_color="on_green")

            # 部分异常不可知，一旦dump失败则回滚数据至上一个采集周期
            self.data_manager.rollback()

    def set_interval(self, interval):
        self.interval = interval

    def inc(self):
        self.data_manager.add_original_data()

        self.count += 1
        if self.count % self.interval:
            return

        self._stop()
        self._dump()
        self._start()
