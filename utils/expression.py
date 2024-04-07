import xml.etree.ElementTree as ET


def expression_build_state(node):
    from core.nodes import Node, ConditionNode
    assert isinstance(node, Node)
    conditions = node.find_nodes(ConditionNode)
    state = { }
    for condition in conditions:
        state[condition.tag()] = False

    return state

def expression_generate_all_possible_states(state):
    keys = list(state.keys())
    num_states = len(keys)

    for i in range(1 << num_states):  # 通过位运算生成所有可能的状态
        possible_state = { }
        for j in range(num_states):
            key = keys[j]
            value = bool((i >> j) & 1)  # 按位检查是否置位
            possible_state[key] = value
        yield possible_state


# 定义运算函数
def expression_evaluate(element, state: dict):
    from core.nodes import Node
    if isinstance(element, str):
        element = ET.fromstring(element)
    elif isinstance(element, Node):
        element = element.to_xml_node()

    tag = element.tag
    if tag in ['And', 'Sequence']:
        return all(expression_evaluate(child, state) for child in element)
    elif tag in ['Or', 'Selector']:
        return any(expression_evaluate(child, state) for child in element)
    elif tag in ['Not', 'Invert']:
        return not expression_evaluate(element[0], state)
    else:  # 检查是否为条件标签
        return state.get(tag, False)
