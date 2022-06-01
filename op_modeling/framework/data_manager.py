import os
import pandas as pd


class ErrorMSG:
    def __init__(self):
        self.dump_dir = None
        self.err_df_file = None
        self.err_origin_info_file = None
        self.err_df = None
        self.err_origin_info = None

    def set_dump_dir(self, dir_):
        self.dump_dir = dir_
        self.err_df_file = os.path.join(self.dump_dir, "err_df.csv")
        self.err_origin_info_file = os.path.join(self.dump_dir, "err_origin_info.csv")

    def set_err_df(self, df):
        self.err_df = df

    def set_err_origin_info(self, info):
        self.err_origin_info = info

    def save(self):
        if self.err_df_file is not None and self.err_df is not None:
            self.err_df.to_csv(self.err_df_file, index=True)
        if self.err_origin_info_file is not None and self.err_origin_info is not None:
            self.err_origin_info.to_csv(self.err_origin_info_file, index=True)
        self.clear()

    def clear(self):
        self.dump_dir = None
        self.err_df_file = None
        self.err_origin_info_file = None
        self.err_df = None
        self.err_origin_info = None


class DataManager:

    def __init__(self, dump_file, already_dump_file=None, insert_op_type=False, insert_op_name=False,
                 dump_call_back=None):

        # dump file
        self.dump_file = dump_file
        self.dump_call_back = dump_call_back

        self.err_msg = ErrorMSG()
        if not os.path.exists(os.path.dirname(dump_file)):
            raise Exception(f"please create {dump_file}")

        if os.path.exists(dump_file):
            df = pd.read_csv(dump_file, keep_default_na=False, na_values=['NA'])
            self.total_prof_infos = df.to_dict('records')
            self.offset = len(self.total_prof_infos)
            print(f"Loaded {self.offset} samples from {dump_file}")
        else:
            self.total_prof_infos = []
            self.offset = 0

        if already_dump_file and os.path.exists(already_dump_file):
            df = pd.read_csv(already_dump_file, keep_default_na=False, na_values=['NA'])
            self.already_prof_infos = df.to_dict('records')
        else:
            self.already_prof_infos = []

        self.cur_input_info = None
        self.op_type = None
        self.op_name = None
        self.op_types = []
        self.op_names = []
        self.insert_op_type = insert_op_type
        self.insert_op_name = insert_op_name

    @staticmethod
    def _check_sample_exists(prof_info_list, original_inputs, attributes):
        for item in prof_info_list:
            if original_inputs == item["Original Inputs"] and attributes == item["Attributes"]:
                return True

        return False

    def find_sample(self, original_inputs, original_outputs, attributes):
        for item in self.total_prof_infos:
            if original_inputs == item["Original Inputs"] and attributes == item["Attributes"]:
                return item

        return None

    def last_sample(self):
        return self.total_prof_infos[-1]

    def check_sample_exists(self, op_type, original_inputs, original_outputs, attributes, op_name=None):
        if self._check_sample_exists(self.total_prof_infos, original_inputs, attributes) \
                or self._check_sample_exists(self.already_prof_infos, original_inputs, attributes):
            return True

        if not op_name:
            op_name = op_type

        self.set_current_info(op_type, op_name, original_inputs, original_outputs, attributes)
        return False

    def set_current_info(self, op_type, op_name, original_inputs, original_outputs, attributes):
        self.cur_input_info = {"Original Inputs": original_inputs,
                               "Original Outputs": original_outputs,
                               "Attributes": attributes}
        if self.insert_op_type:
            self.cur_input_info["OP Type"] = op_type

        if self.insert_op_name:
            self.cur_input_info["OP Name"] = op_name

        self.op_type = op_type
        self.op_name = op_name

    def update_current_info(self, name, value):
        self.cur_input_info[name] = value

    def add_original_data(self):
        self.total_prof_infos.append(dict(self.cur_input_info))
        self.op_types.append(self.op_type)
        self.op_names.append(self.op_name)

    def clear_tempory(self):
        self.op_types.clear()
        self.op_names.clear()

    def dump(self):
        pd.DataFrame(self.total_prof_infos).to_csv(self.dump_file, index=False)

        if self.dump_call_back:
            self.dump_call_back()

        self.clear_tempory()

    def rollback(self):
        """
        该函数用于各种异常情况的回滚，恢复至上一个profiling数据采集周期
        :return:
        """
        # 已恢复
        if len(self.total_prof_infos) == self.offset:
            return
        # 未恢复
        self.total_prof_infos = self.total_prof_infos[:self.offset]
        self.clear_tempory()

    def append(self, prof_df):
        len_prof = len(prof_df)
        len_origin = len(self.total_prof_infos) - self.offset

        if len_prof != len_origin:
            # 保存错误信息，丢弃这次profiling采集
            self.err_msg.set_err_df(prof_df)
            self.err_msg.set_err_origin_info(pd.DataFrame(self.total_prof_infos[self.offset:]))
            # 回滚
            self.rollback()
            raise Exception("Mismatch size for prof data: {} and origin data: {},"
                            " this prof collection will be Discarded"
                            .format(len_prof, len_origin))

        for i in range(len(prof_df)):
            offset = self.offset + i
            self.total_prof_infos[offset].update(dict(prof_df.iloc[i,]))

        # 更新offset
        self.offset += len(prof_df)
