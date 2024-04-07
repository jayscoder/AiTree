from .config import *
from .conditions import *
from .envs import MiniGridSimulation
from core.nodes import register, ActionNode, NODE_STATUS, FAILURE, SUCCESS, Or, Union, And, Invert, Node
import pygame


class MiniGridActionNode(ActionNode):
    @property
    def simulation(self) -> MiniGridSimulation:
        return super().simulation


@register(label='左转')
class TurnLeft(MiniGridActionNode):

    def execute(self) -> NODE_STATUS:
        action = Actions.left
        _, _, _, _, info = self.simulation.step(action=action)
        if info['is_changed']:
            return SUCCESS
        return FAILURE


@register(label='右转')
class TurnRight(MiniGridActionNode):
    def execute(self) -> NODE_STATUS:
        action = Actions.right
        _, _, _, _, info = self.simulation.step(action=action)
        if info['is_changed']:
            return SUCCESS
        return FAILURE


@register(label='前进', props={
    'direction': {
        'type'    : 'string',
        'required': False,
        'options' : DIRECTIONS_OPTIONS,
    }
})
class MoveForward(MiniGridActionNode):
    @property
    def direction(self) -> int:
        return Directions[self.get_prop('direction')]

    def execute(self) -> NODE_STATUS:
        return self.simulation.move_forward(target=self.direction)


@register(label='拾取')
class Pickup(MiniGridActionNode):
    def execute(self) -> NODE_STATUS:
        action = Actions.pickup
        _, _, _, _, info = self.simulation.step(action=action)
        if info['is_changed']:
            return SUCCESS
        return FAILURE


@register(label='放下')
class Drop(MiniGridActionNode):
    def execute(self) -> NODE_STATUS:
        action = Actions.drop
        _, _, _, _, info = self.simulation.step(action=action)
        if info['is_changed']:
            return SUCCESS
        return FAILURE


@register(label='开关')
class Toggle(MiniGridActionNode):
    def execute(self) -> NODE_STATUS:
        action = Actions.toggle
        _, _, _, _, info = self.simulation.step(action=action)
        if info['is_changed']:
            return SUCCESS
        return FAILURE


@register(label='结束')
class Done(MiniGridActionNode):
    def execute(self) -> NODE_STATUS:
        self.simulation.step(Actions.done)
        return SUCCESS


@register(label="手动控制")
class ManualControl(MiniGridActionNode):
    def execute(self) -> NODE_STATUS:
        if not pygame.get_init():
            pygame.init()

        key_pressed = False
        while not key_pressed and not self.simulation.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.context.close()
                    return SUCCESS(msg='Simulation closed')

                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)
                    key_pressed = True

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.simulation.close()
            return

        if key == "backspace":
            self.simulation.node_reset()
            return

        key_to_action = {
            "left"      : Actions.left,
            "right"     : Actions.right,
            "up"        : Actions.forward,
            "space"     : Actions.toggle,
            "pageup"    : Actions.pickup,
            "pagedown"  : Actions.drop,
            "tab"       : Actions.pickup,
            "left shift": Actions.drop,
            "enter"     : Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.simulation.step(action)
        else:
            print(key)


# OpenDoor
@register(
        label='打开门',
        props={
            'color': {
                'type'    : 'string',
                'required': False,
                'options' : COLOR_OPTIONS,
                'desc'    : '门的颜色',
                'visible' : False
            },
            'desc' : """
    目标：打开门
    后置条件：附近的门已经打开
    前置条件：附近1格内有门，且门未打开
    条件为: 附近1格内有门
    """
        })
class OpenDoor(MiniGridActionNode):
    @property
    def color(self) -> str:
        return self.get_prop('color')

    def execute(self) -> NODE_STATUS:

        door = self.simulation.find_nearest_obs(obj='door', color=self.color, near_range=(1, 1))
        if door is None:
            return FAILURE(msg="旁边1格内没有门")

        # 检查门是否打开了
        if door.state == States.open:
            return SUCCESS(msg="门之前已经打开了")

        self.simulation.turn_to(door.pos)
        self.simulation.step(Actions.toggle)

        door = self.simulation.get_obs_item(door.pos)
        # 检查门是否打开了
        if door.state == States.open:
            return SUCCESS(msg="门已打开")
        else:
            return FAILURE(msg="门打开失败")


# PickupKey
@register(label='拾取钥匙', props={
    'color': {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '钥匙的颜色',
        'visible' : False
    },
})
class PickUpKey(MiniGridActionNode):
    @property
    def color(self) -> str:
        return self.get_prop('color')

    def execute(self) -> NODE_STATUS:
        key_obs = self.simulation.find_nearest_obs(obj='key', color=self.color, near_range=(1, 1))
        if key_obs is None:
            return FAILURE(msg="旁边1格内没有钥匙")

        # 先转向
        self.simulation.turn_to(key_obs.pos)
        # 再拾取
        self.simulation.step(Actions.pickup)

        # 检查钥匙是否拾取成功
        key_obs = self.simulation.get_obs_item(key_obs.pos)
        if key_obs.obj != Objects.key:
            return SUCCESS(msg="钥匙已拾取")
        else:
            return FAILURE(msg="钥匙拾取失败")


@register(label="向右移动")
class MoveRight(MoveForward):
    """
    向右移动
    """

    def direction(self) -> int:
        return Directions.right


@register(label="向左移动")
class MoveLeft(MoveForward):
    """
    向左移动
    """

    def direction(self) -> int:
        return Directions.left


@register(label="向上移动")
class MoveUp(MoveForward):
    """
    向前移动
    """

    def direction(self) -> int:
        return Directions.up


@register(label="向下移动")
class MoveDown(MoveForward):
    """
    向下移动
    """

    def direction(self) -> int:
        return Directions.down


@register(label="移动到指定位置", props={
    'x': {
        'type'    : 'int',
        'required': True,
        'desc'    : '目标位置的x坐标'
    },
    'y': {
        'type'    : 'int',
        'required': True,
        'desc'    : '目标位置的y坐标'
    },
})
class MoveToPosition(MiniGridActionNode):
    """
    移动到指定的位置
    """

    @property
    def target_pos(self) -> (int, int):
        return self.get_prop('x'), self.get_prop('y')

    def execute(self) -> NODE_STATUS:
        return self.simulation.move_to(self.target_pos)


@register(label="移动到物体位置", props={
    'object': {
        'type'    : 'string',
        'required': True,
        'options' : OBJECTS_OPTIONS,
        'desc'    : '目标位置的物体'
    },
    'color' : {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '目标位置的物体的颜色',
        'default' : ''
    },
    'nearby': {
        'type'    : 'int',
        'required': False,
        'default' : 0,
        'desc'    : '是否只移动到目标区域旁边n格内',
    },
})
class MoveToObject(MiniGridActionNode):
    """
    移动到指定的目标位置
    """

    @property
    def object(self) -> str:
        return self.get_prop('object')

    @property
    def color(self) -> str:
        return self.get_prop('color')

    @property
    def nearby(self) -> int:
        return self.get_prop('nearby')

    def execute(self) -> NODE_STATUS:
        object_obs = self.simulation.find_nearest_obs(obj=self.object, color=self.color)

        if object_obs is None:
            return FAILURE(msg="找不到目标位置")

        return self.simulation.move_to(object_obs.pos, nearby=self.nearby)


@register(label="移动到物体位置", props={
    'nearby': {
        'type'    : 'int',
        'required': False,
        'default' : 0,
        'desc'    : '是否只移动到目标区域旁边n格内',
        'visible' : False
    },
})
class ApproachObject(MoveToObject):
    """
    移动到指定的目标位置附近1格内
    """

    @property
    def nearby(self) -> int:
        return 1


@register(label="移动到目标位置", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'goal',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '目标位置的物体',
        'visible' : False
    },
})
class MoveToGoal(MoveToObject):
    """
    移动到目标位置
    """

    @property
    def object(self) -> str:
        return 'goal'

    def condition(self) -> Union[Node, None]:
        return CanMoveToGoal()

    def effect(self) -> Union[Node, None]:
        return IsReachGoal()


# MoveToKey
@register(label="移动到钥匙位置附近", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'visible' : False
    },
})
class ApproachKey(ApproachObject):
    """
    移动到钥匙位置
    """

    @property
    def object(self) -> str:
        return 'key'


# MoveToDoor
@register(label="移动到门位置", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'visible' : False
    },
})
class ApproachDoor(ApproachObject):
    """
    移动到门位置
    """

    @property
    def object(self) -> str:
        return 'door'


@register(label="探索未知区域")
class ExploreUnseen(MiniGridActionNode):
    """
    使用AStar算法从当前位置开始探索未知区域
    """

    def condition(self) -> Node:
        return Or(Invert(IsGoalFound), Invert(IsKeyFound), Invert(IsDoorFound))

    def effect(self) -> Node:
        return And(Invert(IsUnseenFound), Or(IsKeyFound, IsDoorFound, IsGoalFound, IsBallFound, IsBoxFound))

    def execute(self) -> NODE_STATUS:
        can_reach_obs = self.simulation.find_can_reach_obs(obj='unseen')
        if can_reach_obs is None:
            # 没有未知区域了
            unseen_obs = self.simulation.find_nearest_obs(obj='unseen')
            if unseen_obs is not None:
                return self.simulation.move_to(unseen_obs.pos)
            return SUCCESS(msg="没有未知区域了")
        return self.simulation.move_to(can_reach_obs.pos)


# TurnToObject
@register(label="转向物体", props={
    'object': {
        'type'    : 'string',
        'required': True,
        'options' : OBJECTS_OPTIONS,
        'desc'    : '目标位置的物体'
    },
    'color' : {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '目标位置的物体的颜色',
        'default' : ''
    },
})
class TurnToObject(MiniGridActionNode):
    """
    转向指定的物体
    """

    @property
    def object(self) -> str:
        return self.get_prop('object')

    @property
    def color(self) -> str:
        return self.get_prop('color')

    def execute(self) -> NODE_STATUS:
        target_obs = self.simulation.find_nearest_obs(obj=self.object, color=self.color)

        if target_obs is None:
            return FAILURE(msg="找不到目标位置")

        return self.simulation.turn_to(target_obs.pos)


# TurnToKey
@register(label="转向钥匙", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'visible' : False
    },
})
class TurnToKey(TurnToObject):
    """
    转向钥匙
    """

    @property
    def object(self) -> str:
        return 'key'


# TurnToDoor
@register(label="转向门", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'visible' : False
    },
})
class TurnToDoor(TurnToObject):
    """
    转向门
    """

    @property
    def object(self) -> str:
        return 'door'
