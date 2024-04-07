import uuid
from typing import Union, Tuple
import xml.etree.ElementTree as ET
import os
import json

import gym.core
import numpy as np


def new_node_id():
    # 只取uuid的前10位
    return uuid.uuid4().hex[:10]


def camel_case_to_snake_case(name):
    """
    驼峰转蛇形
    :param name:
    :return:
    """
    return ''.join(['_' + i.lower() if i.isupper() else i for i in name]).lstrip('_')


def parse_prop_options(options: Union[list, dict, str]) -> list:
    """
    解析属性选项
    :param options:
    :return:
    """
    if isinstance(options, str):
        options = options.split(',')
    if isinstance(options, list):
        result = []
        for option in options:
            if isinstance(option, str):
                result.append({
                    'name' : option,
                    'value': option
                })
            elif isinstance(option, dict):
                result.append({
                    'name' : option.get('name', ''),
                    'value': option.get('value', '')
                })
        return result
    elif isinstance(options, dict):
        result = []
        for key in options:
            result.append({
                'name' : key,
                'value': options[key]
            })
        return result
    return []


PROP_TYPE_MAPPER = {
    'str'   : str,
    'string': str,
    'int'   : int,
    'float' : float,
    'double': float,
    'number': float,
    'bool'  : bool,
    'list'  : list,
    'dict'  : dict,
    'json'  : dict,
}


def parse_prop_type(prop_type: [str, type]):
    if isinstance(prop_type, str):
        prop_type = prop_type.lower()
        if prop_type in PROP_TYPE_MAPPER:
            return PROP_TYPE_MAPPER[prop_type]
        else:
            return str
    else:
        return prop_type


def parse_type_value(value, value_type):
    value_type = parse_prop_type(value_type)
    if value_type == bool:
        return parse_bool_value(value)
    elif value_type == int:
        return parse_int_value(value)
    elif value_type == float:
        return parse_float_value(value)
    elif value_type == str:
        return str(value)
    elif value_type == list:
        return parse_list_value(value)
    elif value_type == dict:
        return parse_dict_value(value)
    elif callable(value_type):
        return value_type(value)
    return value


# 最终props都是以列表的形式保存的
def parse_props(props):
    if props is None:
        return []
    result = []
    if isinstance(props, list):
        for prop in props:
            if isinstance(prop, str):
                result.append({
                    'name'    : prop,
                    'type'    : 'str',
                    'default' : '',
                    'required': False,
                    'desc'    : '',
                    'options' : None,  # 选项 用于下拉框 仅在type为str时有效 {'name': '选项1', 'value': '1'}
                    'visible' : True,  # 是否可见
                })
            elif isinstance(prop, dict):
                result.append({
                    'name'    : prop.get('name', ''),
                    'type'    : prop.get('type', 'str'),
                    'default' : prop.get('default', ''),
                    'required': prop.get('required', False),
                    'desc'    : prop.get('desc', ''),
                    'options' : prop.get('options', None),
                    'visible' : prop.get('visible', True),
                })
    elif isinstance(props, dict):
        for prop in props:
            prop_item = props[prop]
            if isinstance(prop_item, dict):
                result.append({
                    'name'    : prop,
                    'type'    : prop_item.get('type', 'str'),
                    'default' : prop_item.get('default', ''),
                    'required': prop_item.get('required', False),
                    'desc'    : prop_item.get('desc', ''),
                    'options' : prop_item.get('options', None),
                    'visible' : prop_item.get('visible', True),
                })
            elif isinstance(prop_item, type):
                result.append({
                    'name'    : prop,
                    'type'    : prop_item,
                    'default' : '',
                    'required': False,
                    'desc'    : '',
                    'options' : None,
                    'visible' : True,
                })

    for i, item in enumerate(result):
        result[i]['type'] = parse_prop_type(item['type']).__name__
        if not callable(item['default']):
            result[i]['default'] = parse_type_value(value=item['default'], value_type=item['type'])
        result[i]['options'] = parse_prop_options(item['options'])

    return result


def merge_props(props: list, to_props: list):
    """
    合并两个props
    :param props:
    :param to_props:
    :return:
    """
    if to_props is None:
        return props
    to_props = to_props.copy()
    for prop in props:
        find_index = find_prop_index(to_props, prop['name'])
        if find_index == -1:
            to_props.append(prop)
        else:
            to_props[find_index] = prop
    return to_props


def find_prop(meta, name):
    if 'props' in meta:
        for prop in meta['props']:
            if prop['name'] == name:
                return prop
    return None


def find_prop_index(props, name):
    for index, prop in enumerate(props):
        if prop['name'] == name:
            return index
    return -1


def parse_bool_value(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return value > 0
    elif isinstance(value, float):
        return value > 0.0
    elif isinstance(value, str):
        return value.lower() in ['true', '1', 'yes', 'y']
    return False


def parse_int_value(value: str, default: int = 0):
    try:
        return int(value)
    except:
        return default


def parse_float_value(value: str, default: float = 0.0):
    try:
        return float(value)
    except:
        return default


def parse_list_value(value: str, default: list = None):
    try:
        value = json.loads(value)
        return value
    except:
        return default


def parse_dict_value(value: str, default: dict = None):
    try:
        value = json.loads(value)
        return value
    except:
        return default


# 定义一个函数将 XML 元素转换为字典
def xml_to_dict(element):
    result = {
        'tag'       : element.tag,
        'attributes': element.attrib,
        'children'  : [xml_to_dict(child) for child in element]
    }
    return result


def read_xml_to_dict(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return xml_to_dict(root)


# 从目录中提取出所有的xml文件
def extract_xml_files_from_dir(dir_path: str):
    xml_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    return xml_files


# 自定义 JSON 序列化适配器
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint8):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.int8):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        # 对于其他类型的对象，使用 JSONEncoder 的默认处理方式
        return super().default(obj)


def is_obs_same(obs: gym.core.ObsType, other: gym.core.ObsType) -> bool:
    if isinstance(obs, np.ndarray):
        return (obs == other).all()

    if isinstance(obs, gym.core.Dict):
        for key, value in obs.items():
            if not is_obs_same(value, other[key]):
                return False
        return True

    try:
        return obs == other
    except:
        return False
