import shapely.geometry as SG
import numpy as np

start = 0
end = 4000


def levelDivide(index):
    print("当前输入机构健康指数为：", index)
    level = 0
    if index >= 1.0:
        level = 1
    elif 0.8 <= index < 1.0:
        level = 2
    elif 0.5 <= index < 0.8:
        level = 3
    elif index < 0.5:
        level = 4
    return level


# 计算剩余时间函数
def nextLevelTime(level, index, data):
    print("当前输入机构健康等级为：", level)
    border = 0
    if level == 2:
        border = 2900
    elif level == 3:
        border = 4000
    elif level == 1:
        return [float("inf"), -float("inf")]
    indexLine = SG.LineString([(start, index), (end, index)])
    line = SG.LineString(list(zip(data[0], data[1])))
    coords = np.array(line.intersection(indexLine))
    minLife = coords[0][0]
    maxLife = coords[-1][0]
    print("当前输入机构到下一个健康等级的最大和最小时间为：", [border - minLife, border - maxLife])
    return [border - minLife, border - maxLife]
