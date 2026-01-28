# optConfig.py
# 仅包含“用户可配置参数”。不要在此文件中加入任何逻辑代码/函数/导入。

# 多目标数量（cost 向量长度）
NUMBER_OF_OBJECTIVES = 6

# 用户期望的种群规模（会在 misc.py 中修正为 NSGA-III 可接受的 POP_SIZE）
EXPECT_POP_SIZE = 200

# 最大迭代代数
MAX_GENERATION = 50

# 交叉 / 变异概率
CXPB = 1
MUTPB = 1

# SBX 交叉 / 多项式变异参数（DEAP 常用写法）
ETA_CX = 30.0
ETA_MUT = 20.0

# 随机种子；设为 None 表示不固定
RANDOM_SEED = None

# 搜索 NSGA-III 参考点 divisions=P 的上限（通常不用改）
P_CAP = 200

