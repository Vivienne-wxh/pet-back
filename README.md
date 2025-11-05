# 宠物食品安全 AI 问答后端（Python 版）

基于 **FastAPI** 的轻量级后端服务，为前端宠物食品安全助手提供“AI 问答”能力，调用智谱 AI (`glm-4.5-flash`) 模型回答宠物食品与健康相关问题。

## 功能特性

- `POST /ask` 路由，接收 `{ question: string }` 请求体。
- 自动注入专业宠物营养顾问的系统提示词，确保回答风格统一。
- 使用官方 `zai-sdk` 调用智谱开放平台 (`glm-4.5-flash`) 获取回答。
- 控制台输出请求与响应日志，便于调试与监控。
- 默认开启 CORS，前端可直接通过 `fetch` 或 `axios` 访问。

## 快速开始

1. 安装依赖（建议使用虚拟环境）

   ```bash
   pip install -r requirements.txt
   ```

   若只想手动安装 SDK，可执行：

   ```bash
   pip install zai-sdk
   ```

2. 配置环境变量

   在 `backend` 目录下新建 `.env` 文件（可参考 `.env` 模板），写入：

   ```bash
   ZHIPU_API_KEY=你的智谱APIKey
   ```

3. 启动服务

   ```bash
   python app.py
   ```

   启动成功后，控制台打印：

   ```
   ✅ AI 问答接口已启动：http://localhost:3000
   ```

## 接口示例

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"鸡骨头能给小狗吃吗？"}'
```

响应示例：

```json
{
  "answer": "【危险】鸡骨头容易碎裂造成消化道划伤，建议避免喂食。可以选择......"
}
```

## 常见问题

- 若返回 `AI 服务暂时不可用`，请检查 API Key 是否正确、网络是否可访问智谱平台。
- 若控制台提示 `未检测到 ZHIPU_API_KEY`，请确认 `.env` 是否已配置并与 `app.py` 同目录。

## 许可证

MIT License

