from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import utils
from .config import *
from typing import Tuple, Union, List, Dict
from abc import ABC, abstractmethod
from jinja2 import Template
import utils as rl_utils
import time
from collections import defaultdict
from .colors import COLORS
import gymnasium as gym
import numpy as np

_TAG_TO_NODES_CLS = { }  # 用于快速查找节点类，key是register时的tag
_CLASSNAME_TO_NODES_CLS = { }  # 用于快速查找节点类，key是node的类名
_ID_TO_NODES = { }  # 用于快速查找节点，key是node的id

REGISTER_NODES = set()  # 注册的节点


def find_global_node(node_id: str) -> Union[Node, None]:
    """
    从全局查找节点
    :param node_id:
    :return:
    """
    if node_id in _ID_TO_NODES:
        return _ID_TO_NODES[node_id]
    return None


def add_global_node(node: Node):
    """
    添加全局节点
    :param node:
    :return:
    """
    _ID_TO_NODES[node.id] = node


class Context:

    def __init__(self):
        self._blackboard = { }

    @property
    def blackboard(self):
        return self._blackboard

    def __contains__(self, key):
        return key in self._blackboard

    def __getitem__(self, key):
        return self._blackboard[key]

    def __setitem__(self, key, value):
        self._blackboard[key] = value


def register_node(
        node: Union[type, Node],
        **kwargs
):
    """
    注册节点类
    :param node: 节点类或者节点实例
    :return:
    """
    assert callable(node)

    tag = kwargs.get('tag', '')
    props = kwargs.get('props', [])
    if isinstance(tag, str):
        if tag == '':
            if isinstance(node, type):
                tag = [node.__name__]
            elif isinstance(node, Node):
                tag = [node.tag(deep=True)]
            else:
                tag = [str(node)]
        else:
            tag = [tag]

    if node.meta is None:
        node.meta = copy.deepcopy(META_TEMPLATE)

    props = rl_utils.parse_props(props)

    if 'props' in node.meta:
        props = rl_utils.merge_props(props, node.meta['props'])

    node.meta = {
        **node.meta,
        **kwargs,
        'tag'     : tag,
        'props'   : props,
        'visible' : kwargs.get('visible', True),  # 是否在webui上可见，不继承
        'disabled': kwargs.get('disabled', False),  # 是否可用，不继承
        'order'   : kwargs.get('order', 10000),  # 排序，不继承，0表示最前面，-1表示最后面
    }

    all_tags = []
    for _tag in tag:
        _TAG_TO_NODES_CLS[_tag] = node
        all_tags.append(_tag)

    if isinstance(node, type):
        class_name = node.__name__
        snake_class_name = rl_utils.camel_case_to_snake_case(class_name)
        module_name = f'{node.__module__}.{node.__qualname__}'

        _CLASSNAME_TO_NODES_CLS[class_name] = node
        _CLASSNAME_TO_NODES_CLS[snake_class_name] = node
        _CLASSNAME_TO_NODES_CLS[module_name] = node

        for i_tag in [class_name, snake_class_name, module_name]:
            if i_tag not in all_tags:
                all_tags.append(i_tag)

    node.meta['tag'] = all_tags

    REGISTER_NODES.add(node)
    return node


def register(*args, **kwargs):
    """
    注册节点类
    :param args:
        如果只有一个参数，且为函数，则视为装饰器模式，直接注册
        如果有多个参数，则视为普通模式，需要传入tag, desc, props参数
    :param kwargs:
    :return:
    """

    if len(args) == 1 and callable(args[0]):
        return register_node(node=args[0], **kwargs)

    meta = { }
    for arg in args:
        if isinstance(arg, dict):
            meta = { **meta, **arg }

    meta = { **meta, **kwargs }

    node = kwargs.get('node', None)
    if node is None:
        return lambda cls: register_node(node=cls, **meta)

    return register_node(node=node, **meta)


def print_all_node_cls():
    for name in _TAG_TO_NODES_CLS:
        print(name)


def find_node_cls(tag: str, allow_not_found: bool = True) -> Union[type, None]:
    if tag in _TAG_TO_NODES_CLS:
        return _TAG_TO_NODES_CLS[tag]
    elif tag in _CLASSNAME_TO_NODES_CLS:
        return _CLASSNAME_TO_NODES_CLS[tag]
    else:
        return NotFound


@register(tag=['root'], type=NODE_TYPE.VIRTUAL, props={ }, visible=False)
class Node:
    meta = copy.deepcopy(META_TEMPLATE)  # 节点类元数据，用于存储一些额外的信息
    _on_tick = []

    @classmethod
    def add_on_tick(cls, func):
        """
        添加节点tick时的回调函数
        :param func:
        :return:
        """
        cls._on_tick.append(func)
        return func

    @classmethod
    def remove_on_tick(cls, func):
        """
        移除节点tick时的回调函数
        :param func:
        :return:
        """
        cls._on_tick.remove(func)
        return func

    def __init__(self, *args, **kwargs):
        self.children: List[Node] = []

        self._context = kwargs.get('context', Context())  # 节点上下文
        assert isinstance(self._context, Context)

        self.meta = copy.deepcopy(self.__class__.meta)

        for k in kwargs:
            if k in META_TEMPLATE:
                self.meta[k] = kwargs[k]

        self.attributes = copy.deepcopy(kwargs)

        self.cache = { }  # 用于缓存一些节点数据，只有在reset的时候才会清空

        self.status: NODE_STATUS = NODE_STATUS(value=kwargs.get('status', 'not_run'))
        self.tick_count = kwargs.get('tick_count', 0)  # 节点执行次数
        self.success_count = int(kwargs.get('success_count', 0))  # 节点执行成功次数
        self.failure_count = int(kwargs.get('failure_count', 0))  # 节点执行失败次数
        self.running_count = int(kwargs.get('running_count', 0))

        self.id = self.get_prop('id', default=rl_utils.new_node_id())
        self.parent = None
        self.parents = []  # 多个父节点

        self._on_tick = []
        _ID_TO_NODES[self.id] = self

        for arg in args:
            if isinstance(arg, Node):
                self.add_child(arg)
            elif callable(arg):
                self.add_child(arg())
            elif isinstance(arg, NODE_STATUS):
                if arg == SUCCESS:
                    self.add_child(ForceSuccess())
                elif arg == FAILURE:
                    self.add_child(ForceFail())
                elif arg == RUNNING:
                    self.add_child(ForceRunning())
                else:
                    self.add_child(ForceNotRun())

        self._lazy_inited = False  # 延迟初始化，在reset里初始化

    def __call__(self, *args, **kwargs):
        # 用于复制节点
        node = self.__class__(*args, **self.attributes, **kwargs)
        node.context = self.context
        node.meta = copy.deepcopy(self.meta)
        node._on_tick = copy.deepcopy(self._on_tick)
        # 复制所有子节点
        for child in self.children:
            node.add_child(child.__call__())
        return node

    @property
    def label(self):
        """
        节点的标签，用于显示在webui上的节点名称
        """
        return self.get_prop('label') or self.meta.get('label', '')

    @label.setter
    def label(self, value: str):
        self.attributes['label'] = value

    @property
    def desc(self):
        return self.get_prop('desc') or self.meta.get('desc', '')

    @desc.setter
    def desc(self, value: str):
        self.attributes['desc'] = value

    @property
    def disabled(self):
        return self.meta.get('disabled', True)

    @disabled.setter
    def disabled(self, value):
        self.meta['disabled'] = value

    @property
    def visible(self):
        """
        节点是否可见（在webui上）
        """
        return self.meta.get('visible', True)

    @visible.setter
    def visible(self, value):
        """
        设置节点是否可见（在webui上）
        """
        self.meta['visible'] = value

    @property
    def filepath(self):
        fp = self.meta.get('filepath', '')
        if fp == '' and self.parent is not None:
            return self.parent.filepath
        return fp

    @filepath.setter
    def filepath(self, value):
        self.meta['filepath'] = value
        for child in self.children:
            child.filepath = value

    @property
    def sorted_children(self):
        # 返回子节点列表排序后的列表
        return sorted(self.children, key=lambda x: x.__class__.__name__, reverse=True)

    def add_on_tick(self, func):
        """
        添加节点tick时的回调函数
        :param func:
        :return:
        """
        self._on_tick.append(func)
        return

    def remove_on_tick(self, func):
        """
        移除节点tick时的回调函数
        :param func:
        :return:
        """
        self._on_tick.remove(func)
        return

    def replace_child(self, i, child):
        assert isinstance(child, Node), f'child must be Node, but got {type(child)}'
        child.context = self.context
        child.parent = self
        if self not in child.parents:
            child.parents.append(self)
        self.children[i] = child

    def add_child(self, *child):
        for ch in child:
            assert isinstance(ch, Node), f'child must be Node, but got {type(ch)}'
            ch.context = self.context
            ch.parent = self
            if self not in ch.parents:
                ch.parents.append(self)
            self.children.append(ch)
        return self

    def find_node(self, by) -> Union['Node', None]:
        if isinstance(by, str):
            if by == self.id:
                return self
            for child in self.children:
                node = child.find_node(by)
                if node is not None:
                    return node
        elif isinstance(by, type):
            if isinstance(self, by):
                return self
            for child in self.children:
                node = child.find_node(by)
                if node is not None:
                    return node
        elif isinstance(by, list):
            for i_by in by:
                node = self.find_node(i_by)
                if node is not None:
                    return node
        elif isinstance(by, Node):
            return self.find_node(by.id)
        elif callable(by):
            if by(self):
                return self
            for child in self.children:
                node = child.find_node(by)
                if node is not None:
                    return node
        return None

    def find_nodes(self, by) -> List['Node']:
        nodes = []
        if isinstance(by, str):
            if by == self.id:
                nodes.append(self)
            for child in self.children:
                nodes.extend(child.find_nodes(by))
        elif isinstance(by, Node):
            nodes.extend(self.find_nodes(by.id))
        elif isinstance(by, list):
            for i_by in by:
                nodes.extend(self.find_nodes(i_by))
        elif isinstance(by, type):
            if isinstance(self, by):
                nodes.append(self)
            for child in self.children:
                nodes.extend(child.find_nodes(by))
        elif callable(by):
            if by(self):
                nodes.append(self)
            for child in self.children:
                nodes.extend(child.find_nodes(by))
        return nodes

    @property
    def root(self):
        if self.parent is None:
            return self
        return self.parent.root

    @property
    def depth(self):
        if self.parent is None:
            return 1 # 根节点是1层
        return self.parent.depth + 1

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value
        for child in self.children:
            child.context = value

    @property
    def processor(self) -> Processor | None:
        if isinstance(self, Processor):
            return self
        if self.parent is not None:
            return self.parent.processor
        return None

    @property
    def simulation(self) -> Simulation:
        if isinstance(self, Simulation):
            return self
        if self.parent is not None:
            return self.parent.simulation
        return None

    @property
    def simulations(self) -> [Simulation]:
        if isinstance(self, Simulation):
            return [self]

        results = []
        for par in self.parents:
            for sim in par.simulations:
                if sim not in results:
                    results.append(sim)
        return results

    def node_reset(self, context: Context = None):
        if context is not None:
            self._context = context
        self.status = NOT_RUN
        self.cache = { }
        self.tick_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.running_count = 0
        for child in self.children:
            child.node_reset(context=self._context)

        if not self._lazy_inited:
            self._lazy_inited = True
            self.lazy_init()

    def lazy_init(self):
        pass

    def node_count(self) -> int:
        """
        获取节点数量
        :return:
        """
        count = 1
        for child in self.children:
            count += child.node_count()
        return count

    def edge_count(self) -> int:
        """
        获取边数量
        :return:
        """
        count = len(self.children)
        for child in self.children:
            count += child.edge_count()
        return count

    @classmethod
    def module_name(cls):
        return f'{cls.__module__}.{cls.__qualname__}'

    @classmethod
    def from_xml(cls, xml_text, ignore_children=False):
        """
        从xml字符串构建节点
        :param xml_text: xml字符串
        :param ignore_children: 是否忽略子树
        :return:
        """
        xml_root = ET.fromstring(xml_text)
        return cls.from_xml_node(xml_root, ignore_children=ignore_children)

    @classmethod
    def from_xml_node(cls, xml_node, ignore_children=False):
        """
        从xml节点构建节点
        :param xml_node: xml节点
        :param ignore_children: 是否忽略子树
        :return:
        """
        node_cls = find_node_cls(xml_node.tag)
        if node_cls is None:
            raise Exception(f"node class {xml_node.tag} not found")
        node = node_cls(**xml_node.attrib)

        if not ignore_children:
            for child in xml_node:
                node.add_child(cls.from_xml_node(child, ignore_children=ignore_children))
        return node

    @classmethod
    def from_xml_file(cls, filepath: str) -> 'Node':
        if not os.path.exists(filepath):
            raise Exception(f'path: {filepath} not exists')

        files = []
        if os.path.isdir(filepath):
            files.extend(rl_utils.extract_xml_files_from_dir(filepath))
        else:
            files.append(filepath)

        if len(files) == 1:
            xml_root = ET.parse(files[0]).getroot()
            node = Node.from_xml_node(xml_root, ignore_children=False)
            node.filepath = files[0]
        else:
            node = Node()
            node.filepath = filepath
            for path in files:
                child = Node.from_xml_file(path)
                node.add_child(child)
        return node

    def add_xml_node(self, xml_node, filepath: str = '') -> 'Node':
        if find_node_cls(xml_node.tag) is NotFound:
            for item_node in xml_node:
                child = self.from_xml_node(item_node, ignore_children=False)
                child.filepath = filepath
                self.add_child(child)
        else:
            child = self.from_xml_node(xml_node, ignore_children=False)
            child.filepath = filepath
            self.add_child(child)
        return self

    def add_xml_text(self, xml_text: str, filepath: str = '') -> 'Node':
        xml_root = ET.fromstring(xml_text)
        self.add_xml_node(xml_root, filepath=filepath)
        return self

    def add_xml_file(self, *filepath: str) -> 'Node':
        for fp in filepath:
            self.add_child(self.__class__.from_xml_file(fp))
        return self

    ### 主要函数部分 ###

    def condition(self) -> Union[Node, None]:
        """
        行为树原语：前置条件，用于判断当前节点的执行条件
        """
        return None

    def effect(self) -> Union[Node, None]:
        """
        行为树原语：预期执行效果，用于判断当前节点的目标是否达成
        :return:
        """
        return None

    # 度量分数
    def effect_score(self) -> float:
        """
        行为树原语：当前节点的度量分数
        :return:
        """
        eff = self.effect()
        if eff is None:
            return self.status.score
        eff.parent = self
        eff.context = self.context
        return eff.effect_score()

    def skip(self):
        """
        跳过当前节点，当前节点不执行
        :return:
        """
        self.status = NOT_RUN
        for child in self.children:
            child.skip()

    def execute(self) -> NODE_STATUS:
        """
        具体的执行行为，用户不应该直接调用这个方法，而是调用tick方法
        用户应该重写这个方法
        :return:
        """
        status = NOT_RUN
        for child in self.children:
            status = child.tick()
        return status

    @property
    def blackboard(self):
        return self.context.blackboard

    def validate(self):
        """
        验证节点是否合法
        :return: (是否合法, 错误信息)
        """
        if self.id == '' or self.id is None:
            raise ValidateError("id can not be empty")

        # 校验参数
        for prop in self.meta['props']:
            if prop['required']:
                if prop['name'] not in self.attributes:
                    raise ValidateError("prop {} is required".format(prop['name']))
            if prop['type'] != str:
                # 如果prop的类型不是str，则需要校验类型
                value = self.get_prop(prop['name'])
                if not isinstance(value, eval(prop['type'])):
                    raise ValidateError(
                            "prop {} type error, expect {}, but got {}".format(prop['name'], prop['type'], type(value)))
        if self.depth > MAX_TREE_DEPTH:
            raise ValidateError(f'tree depth too large, max depth is {MAX_TREE_DEPTH}')

        # 校验子节点
        for child in self.children:
            child.validate()

    def tick(self) -> NODE_STATUS:
        """
        执行节点
        :return:
        """
        if 'disabled' in self.meta and self.meta['disabled']:
            self.skip()
            return NOT_RUN

        status = self.execute()

        if status is None:
            # 没有返回值的话，视为执行成功
            status = SUCCESS

        self.status = status
        self.tick_count += 1
        if status == SUCCESS:
            self.success_count += 1
        elif status == FAILURE:
            self.failure_count += 1
        elif status == RUNNING:
            self.running_count += 1

        for func in self._on_tick:
            # 执行回调函数，不将cls作为参数传入，而是将self作为参数传入
            func(self)

        for func in self.__class__._on_tick:
            # 执行回调函数，不将cls作为参数传入，而是将self作为参数传入
            func(self)

        return status

    def get_prop(self, name, default=None, once: bool = False):
        """
        获取节点参数
        如果参数值被{{}}包裹，则视为变量，需要从黑板中获取，支持jinja2模板语法
        :param name:
        :param default:
        :param once: 是否只获取一次，如果为True，则只获取一次，之后直接存到cache里
        :return:
        """
        if once and name in self.cache:
            return self.cache[name]

        if isinstance(name, list):
            # 如果name是列表，则视为同一个参数可以有多种表示形式，优先使用靠前的
            for i_name in name:
                value = self.get_prop(name=i_name, default=default, once=once)
                if value is not None:
                    return value
            return None

        prop_rule = rl_utils.find_prop(self.meta, name=name)
        value = self.attributes.get(name, default)
        # if value is None:
        #     value = self.blackboard.get(name, default)

        if value is None and prop_rule is not None:
            value = prop_rule['default']

        if callable(value):
            value = value()

        if prop_rule is not None:
            value = rl_utils.parse_type_value(value, value_type=prop_rule['type'])

        if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
            value = Template(value).render(self.blackboard)

        if prop_rule is not None:
            value = rl_utils.parse_type_value(value, value_type=prop_rule['type'])

        if once:
            self.cache[name] = value

        return value

    def to_xml_node(self, ignore_children: bool = False, sorted_children: bool = False, **kwargs):
        """
        将节点转换为xml节点
        :return:
        """

        attrs = copy.deepcopy(self.attributes)
        if kwargs.get('status', False):
            attrs['status'] = self.status.__str__()
            attrs['tick_count'] = self.tick_count
            attrs['success_count'] = self.success_count
            attrs['failure_count'] = self.failure_count
            attrs['running_count'] = self.running_count

        if kwargs.get('effect_score', False):
            attrs['effect_score'] = self.effect_score()

        if kwargs.get('meta', False):
            attrs = { **attrs, **self.meta }

        if kwargs.get('id', False):
            attrs['id'] = self.id

        if kwargs.get('props', False):
            props = self.meta['props']

            for prop in props:
                attrs[prop['name']] = self.get_prop(prop['name'])

        if kwargs.get('seed', False):
            if self.simulation is not None:
                attrs['seed'] = self.simulation.seed
            kwargs.pop('seed')

        if kwargs.get('filepath', False):
            attrs['filepath'] = self.filepath

        for key in attrs:
            attrs[key] = str(attrs[key])

        node = ET.Element(self.__class__.__name__, attrib=attrs)
        if not ignore_children:
            if sorted_children:
                children = self.sorted_children
            else:
                children = self.children
            for child in children:
                node.append(child.to_xml_node(
                        ignore_children=ignore_children,
                        sorted_children=sorted_children,
                        **kwargs))
        return node

    def to_xml(self, ignore_children: bool = False, sorted_children: bool = False, **kwargs):
        """
        将节点转换为xml字符串
        :return:
        """
        from xml.dom import minidom
        xml_node = self.to_xml_node(
                ignore_children=ignore_children,
                sorted_children=sorted_children,
                **kwargs)
        text = ET.tostring(xml_node, encoding='utf-8').decode('utf-8')
        text = minidom.parseString(text).toprettyxml(indent='    ').replace('<?xml version="1.0" ?>', '').strip()
        return text

    def __str__(self):
        return self.to_xml(ignore_children=True, id=True)

    def __repr__(self):
        return self.__str__()

    def tag(self, deep: bool = False):
        if 'tag' in self.meta:
            tag = self.meta['tag'][0]
        else:
            tag = self.__class__.__name__
        if deep:
            for child in self.children:
                tag += child.tag(deep=deep)
        return tag

    def traverse(self, func):
        """
        遍历节点
        :param func:
        :return:
        """
        func(self)
        for child in self.children:
            child.traverse(func)

    def to_json(self):
        """
        将节点转换为json
        :return:
        """
        node_tags = self.meta.get('tag', [])
        node_label = self.label or self.tag()
        json_data = {
            'id'      : self.id,
            'label'   : node_label,
            'children': [child.to_json() for child in self.children],
            'data'    : {
                'tag'          : node_tags,
                'key'          : node_label,
                'name'         : node_label,
                'label'        : node_label,
                'desc'         : self.desc,
                'filepath'     : self.filepath,
                'tick_count'   : self.tick_count,
                'success_count': self.success_count,
                'failure_count': self.failure_count,
                'running_count': self.running_count,
                'ref_id'       : self.get_prop('ref_id', ''),
                **self.meta,
                **self.attributes,
            }
        }

        json_data['data']['params'] = copy.deepcopy(json_data['data']['props'])
        for param in json_data['data']['params']:
            param['key'] = param['name']
            param['value'] = str(self.get_prop(param['name']))

        return json_data


class NotFound(Node):
    """
    未找到节点，用来作为占位节点使用
    """

    def validate(self):
        super().validate()
        raise ValidateError('node not found')

    def execute(self) -> NODE_STATUS:
        print(f'node {self.label} not found: id={self.id}')
        return NOT_RUN


@register(
        type=NODE_TYPE.PROCESSOR,
        color=COLORS.DEFAULT_PROCESSOR
)
class Processor(Node):
    """
    处理器节点，用于执行行为树，行为树的context会传递给子节点
    每个Processor管理一个新的Context，子节点使用的Context是Processor的Context
    """

    def node_reset(self, context: Context = None):
        super().node_reset(context=context)
        self.context._processor = self

    def execute(self):
        """
        执行
        :return:
        """
        for child in self.children:
            child.tick()

    def processor(self) -> Processor | None:
        return self


@register(type=NODE_TYPE.COMPOSITE, visible=False)
class CompositeNode(Node):
    """
    组合节点
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running_indexes = []

    def node_reset(self, context: Context = None):
        super().node_reset(context=context)
        self.running_indexes = []

    def tick(self) -> NODE_STATUS:
        status = super().tick()
        if status != RUNNING:
            self.running_indexes = []
        return status

    def validate(self):
        """
        组合节点的所有子节点都必须是行为节点/条件节点/组合节点/装饰节点中的其中一个
        :return:
        """
        super().validate()
        for child in self.children:
            if child.meta['type'] not in [
                NODE_TYPE.ACTION,
                NODE_TYPE.CONDITION,
                NODE_TYPE.COMPOSITE,
                NODE_TYPE.DECORATOR,
                NODE_TYPE.SUB_TREE]:
                raise ValidateError(
                        "CompositeNode's child must be ActionNode/ConditionNode/CompositeNode/DecoratorNode/SubTree")
            child.validate()


@register(type=NODE_TYPE.ACTION, visible=False)
class ActionNode(Node):
    """
    行为节点
    """

    def validate(self):
        super().validate()
        if len(self.children) != 0:
            raise ValidateError("ActionNode can not have child node")

    def tick(self) -> NODE_STATUS:
        status = super().tick()
        if DEBUG:
            print(f'{self.tag()} tick: {status} score={status.score} effect_score={self.effect_score()}')
        return status


@register(type=NODE_TYPE.CONDITION, visible=False, color=COLORS.DEFAULT_CONDITION)
class ConditionNode(Node):
    """
    条件节点
    只能向父节点返回 Success或Failed，不得返回运行。
    """

    def validate(self):
        super().validate()
        if len(self.children) != 0:
            raise ValidateError("ConditionNode can not have child node")

    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        :return:
        """
        return self.execute().score


@register(tag=['Sequence', 'And'], order=1, color=COLORS.DEFAULT_SEQUENCE)
class Sequence(CompositeNode):

    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        所有节点分数的平均值
        :return:
        """
        # 只有所有子节点都成功，才算成功
        if len(self.children) == 0:
            return 0
        score = 0
        for child in self.children:
            score += child.effect_score()
        return score / len(self.children)

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            # 没有子节点，直接返回成功
            return SUCCESS

        start_index = 0
        if len(self.running_indexes) > 0:
            # 如果当前存在正在运行列表，则从正在运行列表的第一个节点开始执行
            start_index = self.running_indexes[0]

        status = NOT_RUN

        for index, child in enumerate(self.children):
            if index < start_index:
                child.skip()
                continue

            if status == FAILURE or status == RUNNING:
                # 跳过剩余的节点
                child.skip()
                continue

            child_status = child.tick()
            if child_status == NOT_RUN:
                # 视为直接跳过
                continue

            status = child_status
            if status == RUNNING:
                self.running_indexes = [index]

        return status


@register(tag=['Selector', 'Or'], order=2, color=COLORS.DEFAULT_SELECTOR)
class Selector(CompositeNode):
    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        如果有一个节点分数是1，则返回1，否则返回所有节点分数的平均值
        :return:
        """
        if len(self.children) == 0:
            return 0
        score = 0
        for child in self.children:
            child_score = child.effect_score()
            if child_score == 1:
                return child_score
            score += child_score

        return score / len(self.children)

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            # 没有子节点，直接返回失败
            return FAILURE

        start_index = 0
        if len(self.running_indexes) > 0:
            # 如果当前存在正在运行列表，则从正在运行列表的第一个节点开始执行
            start_index = self.running_indexes[0]

        status = NOT_RUN
        for index, child in enumerate(self.children):
            if index < start_index:
                child.skip()
                continue

            if status == SUCCESS or status == RUNNING:
                # 跳过剩余的节点
                child.skip()
                continue

            child_status = child.tick()
            if child_status == NOT_RUN:
                continue

            status = child_status
            if status == RUNNING:
                self.running_indexes = [index]

        return status


Or = Selector
And = Sequence


@register(
        props={
            'success_threshold': {
                'type'   : 'int',
                'default': 1,
                'desc'   : '成功阈值'
            }
        },
        order=3,
        color=COLORS.DEFAULT_PARALLEL
)
class Parallel(CompositeNode):
    """
    并行节点，依次从头顺次遍历执行所有子节点
    """

    @property
    def success_threshold(self):
        return self.get_prop('success_threshold')

    def effect_score(self) -> float:
        """
        行为树原语：当前节点的衡量分数
        子节点分数从大到小排序，取第success_threshold个
        :return:
        """
        if len(self.children) == 0:
            return 0

        # 只要有一个子节点成功，就算成功
        scores = [s for s in [child.effect_score() for child in self.children]]
        # 从大到小排序
        scores.sort(reverse=True)

        # 如果第success_threshold个节点分数为1，则返回1
        if scores[self.success_threshold - 1] == 1:
            return 1
        # 返回平均值
        return sum(scores) / len(scores)

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            # 没有子节点，直接返回成功
            return SUCCESS

        success_threshold = self.success_threshold
        if success_threshold == -1:
            # 所有子节点都成功才算成功
            success_threshold = len(self.children)

        success_count = 0
        fail_count = 0
        running_indexes = []

        for index, child in enumerate(self.children):
            if len(self.running_indexes) > 0 and index not in self.running_indexes:
                # 如果当前存在正在运行列表 且 当前节点不在正在运行的节点列表中，则跳过
                child.skip()
                continue

            child_status = child.tick()
            if child_status == RUNNING:
                running_indexes.append(index)
            elif child_status == SUCCESS:
                success_count += 1
            elif child_status == FAILURE:
                fail_count += 1

        self.running_indexes = running_indexes
        if success_count >= success_threshold:
            return SUCCESS
        elif len(running_indexes) > 0:
            return RUNNING
        else:
            return FAILURE


@register(color=COLORS.DEFAULT_RANDOM)
class RandomSelector(CompositeNode):
    """
    每次随机一个未执行的节点，总随机次数为子节点个数
    - 当前执行节点返回 SUCCESS，退出停止，向父节点返回 SUCCESS
    - 当前执行节点返回 FAILURE，退出当前节点，继续随机一个未执行的节点开始执行
    - 当前执行节点返回 Running，记录当前节点，向父节点返回 Running，下次执行直接从该节点开始
    - 如果所有节点都返回FAILURE，执行完所有节点后，向父节点返回 FAILURE
    """

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            # 没有子节点，直接返回失败
            return FAILURE

        if len(self.running_indexes) > 0:
            # 如果当前存在正在运行列表，则从正在运行列表的第一个节点开始执行
            select_index = self.running_indexes[0]
            status = NOT_RUN
            for index, child in enumerate(self.children):
                if index != select_index:
                    # 如果当前存在正在运行列表，则继续执行正在运行的节点
                    child.skip()
                    continue
                status = child.tick()
                if status == RUNNING:
                    self.running_indexes = [index]
            return status

        import random
        # shuffle
        indexes = list(range(len(self.children)))
        random.shuffle(indexes)

        status = NOT_RUN

        for index in indexes:
            child = self.children[index]
            if status == SUCCESS or status == RUNNING:
                # 跳过剩余的节点
                child.skip()
                continue

            status = child.tick()
            if status == RUNNING:
                self.running_indexes = [index]

        return status


# register
class RandomSequence(CompositeNode):
    """
    随机顺序执行节点
    """

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            # 没有子节点，直接返回失败
            return FAILURE

        if len(self.running_indexes) > 0:
            # 如果当前存在正在运行列表，则从正在运行列表的第一个节点开始执行
            select_index = self.running_indexes[0]
            status = NOT_RUN
            for index, child in enumerate(self.children):
                if index != select_index:
                    # 如果当前存在正在运行列表，则继续执行正在运行的节点
                    child.skip()
                    continue
                status = child.tick()
                if status == RUNNING:
                    self.running_indexes = [index]
            return status

        import random
        # shuffle
        indexes = list(range(len(self.children)))
        random.shuffle(indexes)

        status = NOT_RUN

        for index in indexes:
            child = self.children[index]
            if status == FAILURE or status == RUNNING:
                # 跳过剩余的节点
                child.skip()
                continue

            status = child.tick()
            if status == RUNNING:
                self.running_indexes = [index]

        return status


# @register
# class RandomWeighted(CompositeNode):
#     """
#     随机权重执行节点
#     """
#
#     def execute(self) -> NODE_STATUS:
#         pass

@register(tag=['Decorator'], type=NODE_TYPE.DECORATOR, color=COLORS.DEFAULT_DECORATOR)
class DecoratorNode(Node):
    """
    装饰节点
    """

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("DecoratorNode must have one child node")

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            return NOT_RUN

        return self.children[0].tick()


@register(tag=['Invert', 'Not'])
class Invert(DecoratorNode):
    """
    取反节点
    """

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("Invert node must have one child node")

    def execute(self) -> NODE_STATUS:
        if len(self.children) == 0:
            return NOT_RUN

        status = self.children[0].tick()
        if status == SUCCESS:
            return FAILURE
        elif status == FAILURE:
            return SUCCESS
        else:
            return status

    def effect_score(self) -> float:
        return 1 - self.children[0].effect_score()


Not = Invert  # 别名


@register
class ForceSuccess(DecoratorNode):
    """
    无条件返回成功节点，执行孩子节点，无论孩子节点返回什么结果，都向父节点返回 SUCCESS
    """

    def execute(self) -> NODE_STATUS:
        for child in self.children:
            child.tick()
        return SUCCESS

    def effect_score(self) -> float:
        return 1


@register
class ForceFail(DecoratorNode):
    """
    无条件返回失败节点
    """

    def execute(self) -> NODE_STATUS:
        for child in self.children:
            child.tick()
        return FAILURE

    def effect_score(self) -> float:
        return 0


@register
class ForceRunning(DecoratorNode):
    """
    无条件返回正在运行节点
    """

    def execute(self) -> NODE_STATUS:
        for child in self.children:
            child.tick()
        return RUNNING

    def effect_score(self) -> float:
        return 0


@register
class ForceNotRun(DecoratorNode):
    """
    无条件返回不运行节点，同时跳过孩子节点
    """

    def execute(self) -> NODE_STATUS:
        for child in self.children:
            child.skip()
        return NOT_RUN

    def effect_score(self) -> float:
        return 0


@register(
        props=[{
            'name'    : 'count',
            'type'    : int,
            'default' : 1,
            'required': True,
            'desc'    : '重复执行节点，最多执行孩子节点count次'
        }]
)
class Repeat(DecoratorNode):
    """
    重复执行节点
    最多执行孩子节点count次，（count作为数据输入），直到孩子节点返回失败，则该节点返回FAILURE，若孩子节点返回RUNNING ，则同样返回RUNNING。
    """

    @property
    def count(self):
        return self.get_prop('count')

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("Repeat node must have one child node")

    def execute(self) -> NODE_STATUS:
        status = NOT_RUN
        for i in range(self.count):
            status = self.children[0].tick()
            if status == FAILURE or status == RUNNING:
                break
        return status


@register
class Retry(DecoratorNode):
    """
    重试节点
    最多执行孩子节点count次，（count作为数据输入），直到孩子节点返回成功，则该节点返回SUCCESS，若孩子节点返回RUNNING ，则同样返回RUNNING。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = kwargs.get('count', 1)

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("Retry node must have one child node")

    def execute(self) -> NODE_STATUS:
        status = NOT_RUN
        for i in range(self.count):
            status = self.children[0].tick()
            if status == SUCCESS or status == RUNNING:
                break
        return status


@register
class UntilFail(DecoratorNode):
    """
    直到失败节点
    执行孩子节点，如果节点返回结果不是 Fail，向父节点返回 Running，直到节点返回 Fail，向父节点返回 Success
    """

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("UntilFail node must have one child node")

    def execute(self) -> NODE_STATUS:
        status = self.children[0].tick()
        if status == FAILURE:
            return SUCCESS
        else:
            return RUNNING


@register
class UntilSuccess(DecoratorNode):
    """
    直到成功节点
    执行孩子节点，如果节点返回结果不是 SUCCESS，向父节点返回 RUNNING，直到节点返回 SUCCESS，向父节点返回 SUCCESS
    """

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise Exception("UntilSuccess node must have one child node")

    def execute(self) -> NODE_STATUS:
        status = self.children[0].tick()
        if status == SUCCESS:
            return SUCCESS
        else:
            return RUNNING


@register(
        props={
            'seconds': {
                'type'   : float,
                'default': 0,
                'desc'   : '节流时间间隔，单位秒'
            },
            'ticks'  : {
                'type'   : int,
                'default': 0,
                'desc'   : '节流tick间隔'
            }
        }
)
class Throttle(DecoratorNode):
    """
    节流节点
    在指定时间内，只执行一次孩子节点（其他时候直接返回上次执行结果），如果孩子节点返回 RUNNING，下次执行时，直接返回 RUNNING
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_execute_time = -1e6
        self.last_execute_tick = -1e6
        self.last_status = NOT_RUN

    @property
    def seconds(self):
        return self.get_prop('seconds')

    def validate(self):
        super().validate()
        if len(self.children) != 1:
            raise ValidateError("Throttle node must have one child node")

    def execute(self) -> NODE_STATUS:
        seconds = self.get_prop('seconds')
        ticks = self.get_prop('ticks')

        if seconds > 0:
            now = time.time()
            if now - self.last_execute_time < seconds:
                return self.last_status
            self.last_execute_time = now
        elif ticks > 0:
            if self.tick_count - self.last_execute_tick < ticks:
                return self.last_status
            self.last_execute_tick = self.tick_count

        self.last_status = self.children[0].tick()
        return self.last_status


# @behavior(key='namespace')
# class NamespaceNode(Parallel):
#     """
#     命名空间节点，执行方式类似于并行节点
#     """
#
#     def __init__(self, name: str, **kwargs):
#         super().__init__(**kwargs)
#         self.name = name
#
#     def validate(self) -> Tuple[bool, str]:
#         """
#         组合节点的所有子节点都必须是行为节点/条件节点/组合节点/装饰节点/变量节点 中的其中一个
#         :return:
#         """
#         for child in self.children:
#             if child.type not in [NODE_TYPE.ACTION, NODE_TYPE.CONDITION, NODE_TYPE.COMPOSITE, NODE_TYPE.DECORATOR,
#                                   NODE_TYPE.VARIABLE]:
#                 return False, "NamespaceNode's child must be ActionNode/ConditionNode/CompositeNode/DecoratorNode/VariableNode"
#             ok, msg = child.validate()
#             if not ok:
#                 return False, msg
#         return True, ''
#
#     def execute(self) -> NODE_STATUS:
#         if len(self.children) == 0:
#             # 没有子节点，直接返回成功
#             return SUCCESS
#
#         success_threshold = self.success_threshold
#         if success_threshold == -1:
#             # 所有子节点都成功才算成功
#             success_threshold = len(self.children)
#
#         success_count = 0
#         fail_count = 0
#         running_indexes = []
#
#         for index, child in enumerate(self.children):
#             if len(self.running_indexes) > 0 and index not in self.running_indexes:
#                 # 如果当前存在正在运行列表 且 当前节点不在正在运行的节点列表中，则跳过
#                 child.skip()
#                 continue
#
#             child_status = child.do_execute()
#             if child_status == RUNNING:
#                 running_indexes.append(index)
#             elif child_status == SUCCESS:
#                 success_count += 1
#             elif child_status == FAILURE:
#                 fail_count += 1
#
#         self.running_indexes = running_indexes
#         if success_count >= success_threshold:
#             return SUCCESS
#         elif len(running_indexes) > 0:
#             return RUNNING
#         else:
#             return FAILURE

@register(type=NODE_TYPE.BEHAVIOR_TREE, order=0, color=COLORS.DEFAULT_BEHAVIOR_TREE)
class BehaviorTree(Parallel):

    @property
    def label(self):
        return self.get_prop('label') or self.id

    @label.setter
    def label(self, value):
        self.attributes['label'] = value


@register(
        label='嵌套节点',
        props={
            'ref_id': {
                'type'    : 'string',
                'default' : '',
                'required': True,
                'desc'    : '嵌套节点引用的BehaviorTree的ID'
            }
        },
        type=NODE_TYPE.SUB_TREE,
        color=COLORS.DEFAULT_SUB_TREE
)
class SubTree(Node):
    """
    子树
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ref_node = None

    @property
    def ref_id(self):
        return self.get_prop('ref_id', once=True)

    @property
    def ref_node(self):
        if self._ref_node is not None:
            return self._ref_node

        ref_id = self.ref_id
        self._ref_node = self.root.find_node(ref_id) or find_global_node(ref_id)
        if self._ref_node is None:
            raise Exception(f"SubTree id {ref_id} not found")
        self._ref_node = self._ref_node(**self.attributes)
        self._ref_node.id = self.id
        self._ref_node.context = self.context

        # 修改所有的子ID, 以防止重复, 所有的子ID都加上父ID作为前缀
        def traverse(node):
            node.id = f'{self.id}|{node.id}'
            for _child in node.children:
                traverse(_child)

        for child in self._ref_node.children:
            traverse(child)

        return self._ref_node

    def validate(self) -> Tuple[bool, str]:
        super().validate()
        if len(self.children) != 0:
            raise ValidateError("SubTree can not have child node")
        ref_id = self.ref_id
        if ref_id == '':
            raise ValidateError(f"SubTree {ref_id}: id can not be empty")
        if ref_id == self.id:
            raise ValidateError(f"SubTree {ref_id}: id can not be self")
        # 引用的节点不能是自己的父亲节点
        par = self.parent
        while par is not None:
            if par.id == ref_id:
                raise ValidateError(f"SubTree {ref_id}: can not reference self parent")
            par = par.parent
        # 校验引用的节点是否存在
        ref_node = self.root.find_node(ref_id) or find_global_node(ref_id)
        if ref_node is None:
            raise ValidateError(f"SubTree {ref_id}: id not found")
        return True, ''

    def execute(self) -> NODE_STATUS:
        if self.ref_node is None:
            return FAILURE
        return self.ref_node.tick()

    def effect_score(self) -> float:
        return self.ref_node.effect_score()


@register(desc='设置黑板变量', props=[
    {
        'name'    : 'key',
        'type'    : 'string',
        'default' : '',
        'required': True,
        'desc'    : '变量名'
    },
    {
        'name'    : 'type',
        'type'    : 'string',
        'default' : 'string',
        'required': False,
        'desc'    : '变量类型'
    },
    {
        'name'    : 'value',
        'type'    : 'string',
        'default' : '',
        'required': False,
        'desc'    : '变量值'
    },
    {
        'name'    : 'once',
        'type'    : 'bool',
        'default' : False,
        'required': False,
        'desc'    : '是否只设置一次'
    }
])
class SetBlackboard(ActionNode):
    """
    设置黑板变量节点
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_set = False

    def validate(self):
        if self.key == '':
            raise ValidateError("key can not be empty")

    def execute(self) -> NODE_STATUS:
        if self.once and self.is_set:
            return SUCCESS
        self.blackboard[self.key] = self.value
        self.is_set = True
        return SUCCESS

    def effect_score(self) -> float:
        return 1

    @property
    def once(self):
        return self.get_prop('once')

    @property
    def key(self):
        return self.get_prop('key')

    @property
    def type(self):
        return self.get_prop('type')

    @property
    def value(self):
        value = self.get_prop('value')
        return rl_utils.parse_type_value(value, value_type=self.type)


@register(desc='打印', props=[{
    "name"   : "msg",
    "type"   : str,
    "desc"   : "要打印的内容",
    "default": ""
}])
class Print(ActionNode):

    @property
    def msg(self):
        return self.get_prop(['msg', 'message'])

    def execute(self) -> NODE_STATUS:
        print(self.msg)
        return SUCCESS

    def effect_score(self) -> float:
        return 1


@register(label='随机动作', desc='随机执行一个仿真动作')
class RandomAction(ActionNode):
    """
    随机执行一个动作
    """

    def execute(self) -> NODE_STATUS:
        _, _, _, _, info = self.simulation.step(action=None)
        if info['is_changed']:
            return SUCCESS
        return FAILURE

    def effect_score(self) -> float:
        return self.status.score


@register
class DoAction(ActionNode):
    """
    执行一个动作
    """

    @property
    def action(self):
        return self.get_prop('action')

    def execute(self) -> NODE_STATUS:
        _, _, _, _, info = self.simulation.step(action=self.action)
        if info['is_changed']:
            return SUCCESS
        return FAILURE

    def effect_score(self) -> float:
        return self.status.score


@register(label='睡眠', desc='睡眠一段时间', props={
    'seconds': {
        'type'   : float,
        'default': 1,
        'desc'   : '睡眠时间，单位秒'
    }
})
class Sleep(ActionNode):
    """
    睡眠节点
    """

    @property
    def seconds(self):
        return self.get_prop('seconds')

    def execute(self) -> NODE_STATUS:
        time.sleep(self.seconds)
        return SUCCESS

    def effect_score(self) -> float:
        return 1


class StepResult:
    def __init__(self, action, obs: dict, reward: float = 0, terminated: bool = False, truncated: bool = False,
                 info: dict = None):
        self.action = action
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info


class Simulation(gym.Wrapper, Node):
    """
    Simulation class
    仿真环境
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        Node.__init__(self)
        self.env_id = env.unwrapped.spec.id
        self.step_count = 0
        self.step_results = []
        self.done = False
        self.seed = 0
        self.train = False
        self.workspace = ''
        self.gif = ''
        self.gif_frames = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.gif_frames = []
        self.seed = seed
        self.step_count = 0
        self.done = False
        obs, info = super().reset(seed=seed, options=options)
        self.step_results = [
            StepResult(
                    action=None,
                    obs=obs,
                    info=info
            )
        ]
        self.node_reset()
        info['is_changed'] = True
        return obs, info

    def step(self, action=None):
        old_obs = self.step_results[-1].obs

        # 执行动作，更新环境，应该是外部调用的接口，这个接口应该尽量别继承
        if self.gif:
            self.gif_frames.append(np.moveaxis(self.get_frame(), 2, 0))

        self.step_count += 1

        if action is None:
            action = self.action_space.sample()
        obs, reward, terminated, truncated, info = super().step(action)
        self.done = terminated or truncated
        self.step_results.append(
                StepResult(action=action, obs=obs, reward=float(reward), terminated=terminated, truncated=truncated,
                           info=info))

        info['is_changed'] = utils.is_obs_same(old_obs, obs)
        return obs, reward, terminated, truncated, info

    def execute(self) -> NODE_STATUS:
        # 执行子节点
        if self.done:
            return NOT_RUN('simulation terminated')
        for node in self.children:
            node.tick()
            if self.done:
                return NOT_RUN('simulation terminated')
        return SUCCESS

    def close(self):
        super().close()
        self.done = True
