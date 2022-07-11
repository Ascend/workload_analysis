import json
from abc import ABCMeta
from abc import abstractmethod

import pandas as pd
from framework.feature_base import FeatureGeneratorBase


class DatasetBase(metaclass=ABCMeta):
    """
    训练测试用数据集的基础抽象，包括能力：
    1. 获取特征
    2. 获取标签
    """

    def get_dataset(self):
        return self.get_features(), self.get_labels()

    @abstractmethod
    def get_features(self):
        raise Exception("This function is not implement")

    @abstractmethod
    def get_labels(self):
        raise Exception("This function is not implement")


class CSVDataset(DatasetBase):
    """
    从落盘的csv文件中读取数据并计算特征和标签
    1. 从自定义保存的内容（Origin Input/Output, Attributes）中获取算子的原始描述
    2. 从profiling数据（Input/Output Shapes, Input/Output Data Types, Input/Output Formats）获取算子最终在芯片上的形态
    """

    def __init__(self, file, label="aicore_time(us)", soc_version="Ascend310P"):
        self.file = file
        self.data = None
        self.fea_ge = None
        self.soc_version = soc_version

        if type(label) is list:
            self.label = label
        else:
            self.label = [label]

    @classmethod
    def parse_origin_inout(cls, desc):
        if desc == 'None':
            return []

        objs = json.loads(desc)
        ret = []
        for obj in objs:
            obj["origin_format"] = obj.pop("format")
            obj["origin_shape"] = obj.pop("shape")
            obj["origin_dtype"] = obj.pop("dtype")
            ret.append(obj)
        return ret

    @classmethod
    def parse_final_inout(cls, shapes, dtypes, formats):
        """
        从profiling数据中解析出最终上板的格式
        """

        def convert_shape(shape_str):
            """
            将字符串描述的shape转化为list
            eg. "1,2,3,4" -> [1,2,3,4]
            """
            if shape_str == "":
                return []
            str_list = shape_str.split(",")
            return [int(dim) for dim in str_list]

        shapes = json.loads(shapes).split(";")
        shapes = [convert_shape(shape) for shape in shapes]
        dtypes = dtypes.split(";")
        # 保持使用小写字母表示dtype
        dtypes = [dtype.lower() for dtype in dtypes]
        formats = formats.split(";")

        ret = []
        for input_shape, dtype, format_ in zip(shapes, dtypes, formats):
            final_desc = {}
            final_desc["shape"] = input_shape
            final_desc["dtype"] = dtype
            final_desc["format"] = format_
            ret.append(final_desc)
        return ret

    @classmethod
    def combine_origin_final(cls, origin_desc, final_desc):
        ret = []
        for origin, final in zip(origin_desc, final_desc):
            desc = {**origin, **final}
            ret.append(desc)

        return ret

    def set_feature_generator(self, fea_ge):
        self.fea_ge = fea_ge

    def get_original_data(self):
        if self.data is None:
            self.data = pd.read_csv(self.file, keep_default_na=False, na_values=['NA'])
        return self.data

    def get_features(self) -> pd.DataFrame:
        if self.data is None:
            self.get_original_data()
        train_data = []
        for _, row in self.data.iterrows():
            inputs, outputs, attributes = self._parse_row(row)
            if self.fea_ge is None:
                raise Exception("Please set feature generate first")
            feature = self.fea_ge.cal_feature(inputs, outputs, attributes)
            train_data.append(feature)

        ret = pd.DataFrame(train_data)
        return ret

    def get_labels(self) -> pd.DataFrame:
        if self.data is None:
            self.get_original_data()
        return self.data[self.label]

    def _parse_row(self, row):
        origin_input_desc = self.parse_origin_inout(row['Original Inputs'])
        origin_output_desc = self.parse_origin_inout(row['Original Outputs'])
        final_input_desc = self.parse_final_inout(row["Input Shapes"], row["Input Data Types"], row["Input Formats"])
        final_output_desc = self.parse_final_inout(row["Output Shapes"], row["Output Data Types"],
                                                   row["Output Formats"])

        input_desc = self.combine_origin_final(origin_input_desc, final_input_desc)
        output_desc = self.combine_origin_final(origin_output_desc, final_output_desc)

        attr = row['Attributes']
        if attr == 'None' or not attr:
            attributes = {}
        else:
            attributes = json.loads(attr)

        return input_desc, output_desc, attributes
