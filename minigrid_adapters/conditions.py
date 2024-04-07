from .config import *
from .envs import MiniGridSimulation
from core.nodes import register, ConditionNode, NODE_STATUS, FAILURE, SUCCESS, Or, Union


class MiniGridConditionNode(ConditionNode):
    @property
    def simulation(self) -> MiniGridSimulation:
        return super().simulation


@register(label="是否发现物体", props={
    'object': {
        'type'    : 'string',
        'required': True,
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体'
    },
    'color' : {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '要查找的物体的颜色',
        'default' : ''
    }
})
class IsObjectFound(MiniGridConditionNode):
    """
    检查是否发现指定的物体
    如果物体颜色为空，则只检查物体类型

    如果物体被自己拾取，则物体的位置和自己的位置相同
    """

    @property
    def object(self):
        return self.get_prop('object')

    @property
    def color(self):
        return self.get_prop('color')

    def execute(self) -> NODE_STATUS:
        memory_obs = self.simulation.memory_obs
        for x in range(memory_obs.shape[0]):
            for y in range(memory_obs.shape[1]):
                object_idx, color_idx, state = memory_obs[x, y, :]
                if object_idx == OBJECT_TO_IDX[self.object] and (
                        self.color == '' or color_idx == COLOR_TO_IDX[self.color]):
                    return SUCCESS
        return FAILURE(msg='Object not found')


# IsUnseenFound
@register(label="是否发现未知区域", props={
    'color': {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '要查找的物体的颜色',
        'default' : '',
        'visible' : False
    }
})
class IsUnseenFound(IsObjectFound):
    """
    是否有未知区域
    """

    @property
    def object(self):
        return 'unseen'


@register(label="是否发现目标", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'goal',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class IsGoalFound(IsObjectFound):
    """
    检查是否发现目标位置
    """

    @property
    def object(self):
        return 'goal'


# IsFoundDoor
@register(label='是否发现门', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'door',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class IsDoorFound(IsObjectFound):
    """
    检查是否发现指定颜色的门
    """

    @property
    def object(self):
        return 'door'


# IsFoundKey
@register(label='是否发现钥匙', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'key',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class IsKeyFound(IsObjectFound):
    """
    检查是否发现指定颜色的钥匙
    """

    @property
    def object(self):
        return 'key'


# IsFoundBall
@register(label='是否发现球', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'ball',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class IsBallFound(IsObjectFound):
    """
    检查是否发现指定颜色的球
    """

    @property
    def object(self):
        return 'ball'


# IsFoundBox
@register(label='是否发现箱子', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'box',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class IsBoxFound(IsObjectFound):
    """
    检查是否发现指定颜色的箱子
    """

    @property
    def object(self):
        return 'box'


@register(label='是否能到达目标位置', props={
    'object': {
        'type'    : 'string',
        'required': True,
        'default' : '',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
    },
    'color' : {
        'type'    : 'string',
        'required': False,
        'default' : '',
        'options' : COLOR_OPTIONS,
        'desc'    : '要查找的物体的颜色',
    }
})
class CanMoveToObject(MiniGridConditionNode):
    """
    检查是否能到达目标位置
    """

    @property
    def object(self):
        return self.get_prop('object')

    @property
    def color(self):
        return self.get_prop('color')

    def condition(self):
        return IsObjectFound(object=self.object, color=self.color)

    def execute(self) -> NODE_STATUS:
        target_obs = self.simulation.find_nearest_obs(obj=self.object, color=self.color)
        if target_obs is None:
            return FAILURE(msg=f'{self.object} not found')

        if self.simulation.can_move_to(target_obs.pos):
            return SUCCESS(msg='Can move to goal')
        else:
            return FAILURE(msg='Cannot move to goal')


# CanMoveToGoal
@register(label='是否能到达目标位置', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'goal',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class CanMoveToGoal(CanMoveToObject):
    """
    检查是否能到达目标位置
    """

    @property
    def object(self):
        return 'goal'

    def condition(self):
        return Or(IsGoalFound(color=self.color), IsDoorOpen(color=self.color))


# CanMoveToGoal
@register(label='是否能到达目标位置', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'goal',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
    'color' : {
        'type'    : 'string',
        'required': False,
        'default' : '',
        'visible' : False
    }
})
class CanMoveToUnseen(CanMoveToObject):
    """
    检查是否能到达未知物体
    """

    @property
    def object(self):
        return 'unseen'

    def condition(self):
        return IsUnseenFound()


# CanApproachDoor
@register(label='是否能接近门', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'door',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class CanApproachDoor(CanMoveToObject):
    """
    检查是否能接近门
    """

    @property
    def object(self):
        return 'door'

    def condition(self):
        return IsDoorFound(color=self.color)


# CanApproachKey
@register(label='是否能接近钥匙', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'key',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class CanApproachKey(CanMoveToObject):
    """
    检查是否能接近钥匙
    """

    @property
    def object(self):
        return 'key'

    def condition(self):
        return IsKeyFound(color=self.color)


# IsReachObject
@register(label='是否到达物体', props={
    'object': {
        'type'    : 'string',
        'required': True,
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体'
    },
    'color' : {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '要查找的物体的颜色',
        'default' : ''
    }
})
class IsReachObject(MiniGridConditionNode):
    """
    检查是否到达指定的物体
    如果物体颜色为空，则只检查物体类型

    如果物体被自己拾取，则物体的位置和自己的位置相同
    """

    @property
    def object(self):
        return self.get_prop('object')

    @property
    def color(self):
        return self.get_prop('color')

    def execute(self) -> NODE_STATUS:
        agent_pos = self.simulation.agent_pos
        agent_obs = self.simulation.get_obs_item(agent_pos)

        if agent_obs is None:
            return FAILURE(msg=f'{self.object} not found')

        if agent_obs.obj == Objects[self.object] and (self.color == '' or agent_obs.color == Colors[self.color]):
            return SUCCESS(msg=f'{self.object} found')

        return FAILURE(msg=f'{self.object} not found')


# IsReachGoal
@register(label='是否到达目标', props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'goal',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '要查找的物体',
        'visible' : False
    },
})
class IsReachGoal(IsReachObject):
    """
    检查是否到达目标位置
    """

    @property
    def object(self):
        return 'goal'

    def condition(self):
        return IsGoalFound(color=self.color)


@register(label="是否靠近物体", props={
    'object'  : {
        'type'    : 'string',
        'required': True,
        'options' : OBJECTS_OPTIONS,
        'desc'    : '目标位置的物体'
    },
    'color'   : {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '目标位置的物体的颜色',
        'default' : ''
    },
    'distance': {
        'type'    : 'int',
        'required': False,
        'default' : 1,
        'desc'    : '曼哈顿距离',
    }
})
class IsNearObject(MiniGridConditionNode):
    """
    检查自己是否在目标物体位置附近，如果distance为0，则检查是否在目标位置
    """

    @property
    def object(self) -> str:
        return self.get_prop('object')

    @property
    def color(self) -> str:
        return self.get_prop('color')

    @property
    def distance(self) -> int:
        return self.get_prop('distance')

    def condition(self):
        return IsObjectFound(object=self.object, color=self.color)

    def execute(self) -> NODE_STATUS:
        find_obj = self.simulation.find_nearest_obs(obj=self.object, color=self.color,
                                                    near_range=(self.distance, self.distance))
        if find_obj is None:
            return FAILURE(msg=f'附近找不到{self.object}')
        return SUCCESS(msg=f'在{self.object}位置附近')


@register(label="是否靠近目标", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'goal',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '目标位置的物体',
        'visible' : False
    },
})
class IsNearGoal(IsNearObject):
    """
    检查自己是否在目标位置
    """

    @property
    def object(self) -> str:
        return 'goal'


@register(label="是否在门附近", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'door',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '门',
        'visible' : False
    },
})
class IsNearDoor(IsNearObject):
    """
    检查自己是否在门前
    """

    @property
    def object(self) -> str:
        return 'door'

    def condition(self):
        return IsDoorFound(color=self.color)


@register(label="是否在钥匙附近", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'key',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '钥匙',
        'visible' : False
    },
})
class IsNearKey(IsNearObject):
    """
    检查自己是否在钥匙附近
    """

    @property
    def object(self) -> str:
        return 'key'

    def condition(self):
        return IsKeyFound(color=self.color)


@register(label="是否在球附近", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'ball',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '球',
        'visible' : False
    },
})
class IsNearBall(IsNearObject):
    """
    检查自己是否在球附近
    """

    @property
    def object(self) -> str:
        return 'ball'

    def condition(self):
        return IsBallFound(color=self.color)


# IsNearBox
@register(label="是否在箱子附近", props={
    'object': {
        'type'    : 'string',
        'required': False,
        'default' : 'box',
        'options' : OBJECTS_OPTIONS,
        'desc'    : '箱子',
        'visible' : False
    },
})
class IsNearBox(IsNearObject):
    """
    检查自己是否在箱子附近
    """

    @property
    def object(self) -> str:
        return 'box'

    def condition(self):
        return IsBoxFound(color=self.color)


# 是否持有钥匙
@register(label="是否持有钥匙", props={
    'object': {
        'type'   : 'str',
        'visible': False,
    },
})
class IsKeyHeld(IsReachObject):
    """
    检查自己是否持有钥匙
    """

    @property
    def object(self) -> str:
        return 'key'

    def condition(self):
        return IsKeyFound(color=self.color)


# IsObjectInFront
@register(label="物体是否在正前方", props={
    'object'  : {
        'type'    : 'string',
        'required': True,
        'options' : OBJECTS_OPTIONS,
        'desc'    : '目标位置的物体'
    },
    'color'   : {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '目标位置的物体的颜色',
        'default' : ''
    },
    'distance': {
        'type'    : 'int',
        'required': False,
        'default' : 1,
        'desc'    : '曼哈顿距离',
    }
})
class IsObjectInFront(MiniGridConditionNode):
    """
    检查自己是否在物体正前方
    """

    @property
    def object(self) -> str:
        return self.get_prop('object')

    @property
    def color(self) -> str:
        return self.get_prop('color')

    def condition(self):
        return IsObjectFound(object=self.object, color=self.color)

    def execute(self) -> NODE_STATUS:
        front_pos = self.simulation.front_pos
        find_obj = self.simulation.get_obs_item(front_pos)
        if find_obj is None:
            return FAILURE(msg=f'正前方找不到{self.object} {self.color}')
        if find_obj.obj == Objects[self.object] and (self.color == '' or find_obj.color == Colors[self.color]):
            return SUCCESS(msg=f'在{self.object}正前方 {self.color}')
        return FAILURE(msg=f'正前方找不到{self.object} {self.color}')


# IsKeyInFront
@register(label="钥匙是否在正前方", props={
    'object': {
        'type'   : 'str',
        'visible': False,
    },
})
class IsKeyInFront(IsObjectInFront):
    """
    钥匙是否在正前方
    """

    @property
    def object(self) -> str:
        return 'key'

    def condition(self):
        return IsKeyFound(color=self.color)


# IsDoorInFront
@register(label="门是否在正前方", props={
    'object': {
        'type'   : 'str',
        'visible': False,
    },
})
class IsDoorInFront(IsObjectInFront):
    """
    门是否在正前方
    """

    @property
    def object(self) -> str:
        return 'door'

    def condition(self):
        return IsDoorFound(color=self.color)


# IsDoorOpen
@register(label='最近的门是否打开', props={
    'color': {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '门的颜色',
        'visible' : False
    },
})
class IsDoorOpen(MiniGridConditionNode):

    @property
    def color(self) -> str:
        return self.get_prop('color')

    def condition(self) -> Union[ConditionNode, None]:
        return IsDoorFound(color=self.color)

    def execute(self) -> NODE_STATUS:
        # 找到离自己最近的门
        door = self.simulation.find_nearest_obs(obj='door', color=self.color)
        if door is None:
            return FAILURE(msg="附近找不到门")
        # 检查门是否打开了
        if door.state == States.open:
            return SUCCESS(msg="门已打开")
        else:
            return FAILURE(msg="门未打开")


# IsDoorClosed
@register(label='最近的门是否关闭', props={
    'color': {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '门的颜色',
        'visible' : False
    },
})
class IsDoorClosed(MiniGridConditionNode):

    @property
    def color(self) -> str:
        return self.get_prop('color')

    def condition(self) -> Union[ConditionNode, None]:
        return IsDoorFound(color=self.color)

    def execute(self) -> NODE_STATUS:
        # 找到离自己最近的门
        door = self.simulation.find_nearest_obs(obj='door', color=self.color)
        if door is None:
            return FAILURE(msg="附近找不到门")
        # 检查门是否关闭了
        if door.state == States.closed:
            return SUCCESS(msg="门已关闭")
        else:
            return FAILURE(msg="门未关闭")


# IsDoorLocked
@register(label='最近的门是否锁定', props={
    'color': {
        'type'    : 'string',
        'required': False,
        'options' : COLOR_OPTIONS,
        'desc'    : '门的颜色',
        'visible' : False
    },
})
class IsDoorLocked(MiniGridConditionNode):

    @property
    def color(self) -> str:
        return self.get_prop('color')

    def condition(self) -> Union[ConditionNode, None]:
        return IsDoorFound(color=self.color)

    def execute(self) -> NODE_STATUS:
        # 找到离自己最近的门
        door = self.simulation.find_nearest_obs(obj='door', color=self.color)
        if door is None:
            return FAILURE(msg="附近找不到门")
        # 检查门是否锁定了
        if door.state == States.locked:
            return SUCCESS(msg="门已锁定")
        else:
            return FAILURE(msg="门未锁定")
