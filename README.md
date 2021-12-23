# Optimization_Project

## 开发规则
1. 各个模块用 `*.py` 存在根目录
2. 各个模块自行测试，准备调用 example 在 `if __name__ == 'main.py':`
3. 各个模块包含足够满足相应需求的函数
4. 整合在 `optimize.ipynb`
5. **控制自己不要在 numpy 中使用for循环**

## 进度规划
1. 先使用自己生成的随机数把代码跑通
2. 扩展到给出的数据转稠密在少量样本下能跑通
3. 研究如何使用稀疏矩阵跑通

## 目录结构
- `data` 文件夹，存放数据
- `gen.py` 随机数生成
- `load_data.py` 加载指定数据
- `plot.py` 绘图相关
- `AGM.py` 加速梯度法
- `Newto_CG.py` 牛顿CG法
- `func_tools.py` 和损失函数相关的辅助函数（类似求梯度等）

## 进度一览
- [x] 数据存取
- [ ] 辅助函数
- [x] AGM
- [ ] ...