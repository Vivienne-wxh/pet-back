# 后端部署说明 - Render

## 环境变量配置

部署到 Render 后，**必须配置以下环境变量**：

### 必需的环境变量

1. **ZHIPU_API_KEY**
   - 从 [智谱AI开放平台](https://open.bigmodel.cn/) 获取
   - 在 Render 项目设置中添加：
     - Key: `ZHIPU_API_KEY`
     - Value: `你的智谱AI API Key`

### 配置步骤

1. 登录 Render 控制台
2. 进入你的服务（pet-back）
3. 点击左侧 "Environment" 菜单
4. 点击 "Add Environment Variable"
5. 添加：
   - Key: `ZHIPU_API_KEY`
   - Value: 你的 API Key（从智谱AI平台获取）
6. 保存后，Render 会自动重新部署

### 验证配置

部署成功后，访问：
- `https://pet-back-zk67.onrender.com/` - 应该返回服务信息
- `https://pet-back-zk67.onrender.com/docs` - 应该显示 API 文档

### 常见问题

**问题：AI 服务返回"暂时不可用"**
- 原因：未配置 `ZHIPU_API_KEY` 环境变量
- 解决：按照上述步骤配置环境变量

**问题：部署失败**
- 检查 `requirements.txt` 是否包含所有依赖
- 检查 `Procfile` 是否正确
- 查看 Render 日志排查具体错误

