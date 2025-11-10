from .conv3d import SimpleConv3D, BasicConv3D

__all__ = ['SimpleConv3D', 'BasicConv3D']

def create_model(model_name='simple', **kwargs):
    """创建模型工厂函数"""
    models = {
        'simple': SimpleConv3D,
        'basic': BasicConv3D
    }
    
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}，可选: {list(models.keys())}")
    
    return models[model_name](**kwargs)