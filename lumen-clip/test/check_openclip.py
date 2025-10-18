import open_clip

# 列出所有支持的模型架构
models = open_clip.list_models()
print(f"Total models: {len(models)}")
print(models)

# 检查特定模型是否存在
print("bioclip-2" in models)
