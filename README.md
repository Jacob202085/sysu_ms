# sysu_ms
I'm trying to submit some code to get familiar with this site again.

conv3d_project/

├── data/           # 数据相关模块
│   ├── __init__.py
│   ├── datasets.py # 数据集定义 (如 Synthetic3DDataset)
│   ├── generators.py # 数据生成器 (如 DataGenerator)
│   ├── loaders.py  # 数据加载工具 (如 create_dataloader)
│   └── transforms.py # 数据预处理变换 (如 get_train_transforms)
├── models/         # 模型定义
│   ├── __init__.py
│   └── conv3d.py   # 3D卷积网络模型 (如 SimpleConv3D)
├── utils/          # 工具函数
│   ├── __init__.py
│   └── config.py   # 配置管理和参数解析
├── train.py        # 模型训练脚本
├── test.py         # 模型测试脚本
├── requirements.txt # 项目依赖
└── README.md       # 项目说明文档
