data_type = {
    "float": 0,  # vector运行,开发明确告知x1、x2只支持一个转置
    "float16": 1,  # cube运行，支持转置，其他类型会报错
    "int8": 2,  # cube运行,开发明确告知不支持转置
    "int32": 3,  # vector运行,开发明确告知x1、x2只支持一个转置
}

max_value = 1000
