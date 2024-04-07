import matplotlib.pyplot as plt

# 创建示例的回报列表（可以根据实际情况替换为你自己的数据）
return_per_episode = [0, 10, 20, 30, 25, 40, 35, 50, 45, 60]


def plot_and_save(x, y, title='', xlabel='', ylabel='', savefig: str = ''):
    # 创建x轴的数据（每一集的序号）
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(x)  # 设置x轴刻度为每一集的序号

    # 保存图像到文件
    plt.savefig(savefig)
