"""app.py
------------------------------
该文件使用 FastAPI 构建宠物食品安全 AI 问答后端服务。
"""

# 引入标准库与第三方库
# - os: 读取系统环境变量
# - logging: 统一日志输出，方便调试
# - typing.Optional, typing.Any: 类型注解
# - fastapi: 创建 Web 服务
# - fastapi.middleware.cors: 处理跨域请求
# - fastapi.responses: 流式响应支持
# - pydantic: 定义请求体数据模型
# - zai: 官方智谱 AI Python SDK，封装模型调用
# - dotenv: 从 .env 文件加载环境变量
import os
import json
import logging
import time
from typing import Optional, Any, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from zai import ZhipuAiClient


# 从 .env 文件加载环境变量，确保 ZHIPU_API_KEY 在运行前已正确配置。
load_dotenv()


# 配置日志格式与等级，使控制台可以输出详细的请求/响应信息。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# 初始化 FastAPI 应用实例。
app = FastAPI(
    title="宠物食品安全 AI 问答后端",
    description="提供 POST /ask 接口，通过智谱 AI glm-4 回答宠物食品与健康问题。",
    version="1.0.0"
)


# 配置 CORS 中间件，允许前端在本地或其他域名下直接访问。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可根据需要限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义请求体数据模型
class PetProfile(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    breed: Optional[str] = None
    allergies: Optional[list[str]] = None

class AskRequest(BaseModel):
    question: str
    pet_profile: Optional[PetProfile] = None  # 可选的宠物档案信息


# 智谱 AI 配置常量。

def build_system_prompt(pet_profile: Optional[PetProfile] = None) -> str:
    """构建系统提示词，根据宠物档案信息动态生成。
    
    Args:
        pet_profile: 宠物档案信息
        
    Returns:
        系统提示词字符串
    """
    # 优化：简化提示词，减少token数，加快响应
    base_prompt = "你是宠物营养专家。简要回答。\n重要：只输出最终答案，不要输出思考过程、推理过程或任何标签（如<thinking>、<reasoning>等）。"
    
    # 如果有宠物档案，添加过敏原检查指令
    if pet_profile and pet_profile.allergies and len(pet_profile.allergies) > 0:
        pet_name = pet_profile.name or "该宠物"
        allergies_str = "、".join(pet_profile.allergies)
        base_prompt += f"\n过敏原：{pet_name}对{allergies_str}过敏。如食物含过敏原，标记【高危预警】，禁止喂食。"
    
    base_prompt += "\n格式：\n【风险等级】：[等级]\n【风险点】：[风险]\n【喂养建议】：[建议]"
    
    return base_prompt


ZHIPU_MODEL_NAME: str = "GLM-4-Flash-250414"

# app.py - 在 ZHIPU_MODEL_NAME 常量下方加入
# ---------------------------------------------------------------------------
# RAG 知识库
# ---------------------------------------------------------------------------
PET_KNOWLEDGE = {
    "巧克力": "【高危预警】含可可碱，对宠物有毒，剂量大有生命危险。",  
    "葡萄": "【高危预警】可能导致肾衰竭，少量也危险。建议立即就医。", 
    "洋葱": "【高危预警】破坏红细胞，引起贫血，剂量大有生命危险。",  
    "木糖醇": "【高危预警】导致胰岛素大量分泌，引起低血糖、肝衰竭，对宠物极度危险。",
    "牛油果": "【中危预警】含毒性物质persin，虽然狗猫反应较小，但不建议食用。",
    "生鸡蛋": "【中危预警】可能含有沙门氏菌，应煮熟，长期食用生蛋白会影响生物素吸收。",  
    "咖啡": "【高危预警】含咖啡因，可能引起中毒。",
    "茶": "【高危预警】含咖啡因。",
    "西瓜": "【低风险】少量果肉安全，但种子和瓜皮不宜食用，糖尿病宠物需谨慎。",
    "苹果": "【低风险】果肉安全，但果核含有氰化物，必须去除。",
}


def get_rag_info(question: str) -> str:
    """RAG 知识检索：通过关键词匹配，从本地知识库中检索安全信息。"""
    retrieved_facts = []
    lower_question = question.lower()

    # 优化：使用更高效的匹配方式，只匹配第一个匹配项（大多数情况下只需要一个）
    for item, fact in PET_KNOWLEDGE.items():
        if item in lower_question:
            retrieved_facts.append(f"【{item}】{fact}")
            break  # 找到第一个匹配就返回，减少处理时间

    if retrieved_facts:
       # 进一步简化格式，减少token数
       return "\n参考：" + retrieved_facts[0]
    else:
        return ""
# ---------------------------------------------------------------------------

import re

def filter_thinking_content(content: str) -> str:
    """过滤思考过程内容，移除思考过程相关的标签和文本。
    
    Args:
        content: 原始内容
        
    Returns:
        过滤后的内容
    """
    if not content:
        return content
    
    # 移除思考过程标签（成对标签，如<thinking>...</thinking>）
    content = re.sub(r'<[^>]*(?:thinking|reasoning|redacted)[^>]*>.*?</[^>]*(?:thinking|reasoning|redacted)[^>]*>', '', content, flags=re.DOTALL | re.IGNORECASE)
    # 移除自闭合标签（如<thinking/>）
    content = re.sub(r'<[^>]*(?:thinking|reasoning|redacted)[^>]*/?>', '', content, flags=re.IGNORECASE)
    
    return content

def format_ai_response(text: str) -> str:
    """格式化AI回答，确保【风险等级】、【风险点】、【喂养建议】独立成行。
    
    Args:
        text: 原始AI回答文本
        
    Returns:
        格式化后的文本
    """
    if not text:
        return text
    
    # 移除文本首尾空白
    text = text.strip()
    
    # 过滤思考过程：移除所有思考过程相关的标签和内容
    # 匹配各种可能的思考过程标签，包括<thinking>、<reasoning>、<think>等
    text = re.sub(r'<[^>]*(?:thinking|reasoning|redacted)[^>]*>.*?</[^>]*(?:thinking|reasoning|redacted)[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 匹配自闭合标签
    text = re.sub(r'<[^>]*(?:thinking|reasoning|redacted)[^>]*/?>', '', text, flags=re.IGNORECASE)
    # 移除可能残留的思考过程内容（如果标签被移除但内容还在）
    text = re.sub(r'思考过程[：:].*?(?=【|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'推理过程[：:].*?(?=【|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 第一步：统一冒号格式（英文冒号改为中文冒号）
    text = re.sub(r'【风险等级】:\s*', '【风险等级】：', text)
    text = re.sub(r'【风险点】:\s*', '【风险点】：', text)
    text = re.sub(r'【喂养建议】:\s*', '【喂养建议】：', text)
    
    # 第二步：最关键 - 强制在标题之间插入换行（更全面的匹配）
    # 使用更宽泛的模式，匹配任意字符直到下一个标题
    
    # 处理【风险等级】和【风险点】之间的换行
    # 匹配：【风险等级】后面任意内容直到【风险点】
    text = re.sub(r'【风险等级】[:：]?([^【]*?)【风险点】', r'【风险等级】：\1\n\n【风险点】', text, flags=re.DOTALL)
    
    # 处理【风险点】和【喂养建议】之间的换行
    text = re.sub(r'【风险点】[:：]?([^【]*?)【喂养建议】', r'【风险点】：\1\n\n【喂养建议】', text, flags=re.DOTALL)
    
    # 第四步：统一标题后的冒号格式（确保都有中文冒号）
    text = re.sub(r'【风险等级】\s*[:：]\s*', '【风险等级】：', text)
    text = re.sub(r'【风险点】\s*[:：]\s*', '【风险点】：', text)
    text = re.sub(r'【喂养建议】\s*[:：]\s*', '【喂养建议】：', text)
    
    # 第五步：重新组织文本，确保每个部分（标题+内容）之间有空行
    lines = text.split('\n')
    formatted_lines = []
    current_section = []  # 当前正在处理的章节
    section_title = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检查是否是新的标题行
        is_header = False
        header_type = None
        if '【风险等级】' in line:
            is_header = True
            header_type = '风险等级'
        elif '【风险点】' in line:
            is_header = True
            header_type = '风险点'
        elif '【喂养建议】' in line:
            is_header = True
            header_type = '喂养建议'
        
        if is_header:
            # 如果之前有章节，先保存它（在章节后添加空行）
            if current_section:
                formatted_lines.extend(current_section)
                formatted_lines.append('')  # 章节之间添加空行
            
            # 开始新章节
            current_section = []
            section_title = header_type
            # 确保标题格式正确
            line = re.sub(r'【风险等级】\s*[:：]?\s*', '【风险等级】：', line)
            line = re.sub(r'【风险点】\s*[:：]?\s*', '【风险点】：', line)
            line = re.sub(r'【喂养建议】\s*[:：]?\s*', '【喂养建议】：', line)
            current_section.append(line)
        else:
            # 当前行的内容属于当前章节
            if current_section:
                current_section.append(line)
            else:
                # 如果没有当前章节，直接添加（理论上不应该发生）
                formatted_lines.append(line)
    
    # 添加最后一个章节
    if current_section:
        formatted_lines.extend(current_section)
    
    text = '\n'.join(formatted_lines)
    
    # 第六步：清理多余的空行（超过2个连续换行变为2个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 第七步：确保文本开头和结尾格式正确（但保留各章节之间的空行）
    # 只移除开头和结尾的空行，保留中间的空行
    text = text.strip()
    
    return text


_zhipu_client: Optional[ZhipuAiClient] = None

def stream_zhipu_ai_response(question: str, pet_profile: Optional[PetProfile] = None) -> Generator[str, None, None]:
    """流式调用智谱 AI 接口，实时返回文本块。
    
    Args:
        question (str): 用户提出的宠物食品或健康问题。
        pet_profile: 可选的宠物档案信息，包含过敏原等
    
    Yields:
        str: JSON格式的数据块，包含content字段。
    """
    start_time = time.time()
    
    # 用于收集完整内容，以便在最后进行格式化
    full_content_buffer = []
    api_key: Optional[str] = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        error_msg = json.dumps({"error": "服务器未配置 AI 服务，请联系管理员。"}, ensure_ascii=False)
        yield f"data: {error_msg}\n\n"
        return

    # 优化：立即发送"思考中"状态，让前端立即知道请求已收到（优化首字节时间）
    # 这样即使后续处理慢，用户也能立即看到响应
    yield f"data: {json.dumps({'status': 'thinking'}, ensure_ascii=False)}\n\n"

    global _zhipu_client
    if _zhipu_client is None:
        try:
            _zhipu_client = ZhipuAiClient(api_key=api_key)
        except Exception as exc:
            logger.error("❌ 初始化智谱 AI 客户端失败：%s", exc)
            error_msg = json.dumps({"error": "AI 客户端初始化失败，请检查配置。"}, ensure_ascii=False)
            yield f"data: {error_msg}\n\n"
            return

    client = _zhipu_client

    # 1. RAG 检索：根据用户问题从知识库中获取相关信息（优化：快速检索，减少开销）
    rag_context = get_rag_info(question)

    # 2. 根据宠物档案构建动态系统提示词（优化：简化提示词长度）
    system_prompt = build_system_prompt(pet_profile)
    
    # 3. 构造完整的用户 Prompt：原始问题 + RAG 检索结果
    full_user_prompt = question
    if rag_context:
        full_user_prompt += rag_context

    # 4. 构造 messages 列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user_prompt},
    ]

    try:
        
        # 启用流式传输
        # 优化参数以提升速度：降低temperature、减少max_tokens、优化模型参数
        response_stream = client.chat.completions.create(
            model=ZHIPU_MODEL_NAME,
            messages=messages,
            stream=True,
            max_tokens=600,  # 设置为600，确保三个部分都能完整输出
            temperature=0.1,  # 进一步降低到0.1，加快响应速度
            top_p=0.8,  # 添加top_p参数，加快采样速度
        )

        first_chunk_time = None
        # 流式返回每个数据块 - 直接迭代，立即发送
        for chunk in response_stream:
            # 处理不同类型的chunk（dict或对象）
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [])
            elif hasattr(chunk, "choices"):
                choices = getattr(chunk, "choices", [])
            else:
                # 尝试转换为dict
                choices = []
                if hasattr(chunk, "__dict__"):
                    chunk_dict = chunk.__dict__
                    choices = chunk_dict.get("choices", [])
            
            if choices and len(choices) > 0:
                choice = choices[0]
                # 提取delta
                if isinstance(choice, dict):
                    delta = choice.get("delta", {})
                elif hasattr(choice, "delta"):
                    delta = getattr(choice, "delta", {})
                else:
                    delta = {}
                
                # 明确忽略思考过程（reasoning_content），只处理实际内容（content）
                # 根据智谱AI文档，思考过程通过 reasoning_content 传递，实际内容通过 content 传递
                # 一个chunk可能同时包含 reasoning_content 和 content，我们只处理 content
                if isinstance(delta, dict):
                    # 只提取 content，忽略 reasoning_content
                    content = delta.get("content")
                elif hasattr(delta, "content"):
                    content = getattr(delta, "content", None)
                else:
                    content = None
                
                # 如果这个chunk只有 reasoning_content 而没有 content，跳过
                if not content:
                    continue
                
                if content:
                    # 实时过滤思考过程内容
                    filtered_content = filter_thinking_content(content)
                    
                    # 如果过滤后内容为空，跳过这个chunk
                    if not filtered_content:
                        continue
                    
                    # 记录首字节时间（仅第一次）
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        elapsed = first_chunk_time - start_time
                        logger.info(f"⚡ 首字节时间: {elapsed:.3f}s")
                    
                    # 收集内容用于最终格式化（使用原始内容，在最后统一过滤）
                    full_content_buffer.append(content)
                    # 立即发送过滤后的内容块，实现真正的逐字流式传输
                    data = json.dumps({"content": filtered_content}, ensure_ascii=False)
                    yield f"data: {data}\n\n"

        # 流式传输完成后，进行格式化处理（异步处理，不阻塞流式输出）
        if full_content_buffer:
            full_text = ''.join(full_content_buffer)
            
            # 只在调试模式下记录详细日志（减少日志开销）
            if logger.level <= logging.DEBUG:
                logger.debug(f"📝 原始AI回答（前200字符）: {repr(full_text[:200])}")
            
            # 检查是否包含三个必要部分
            has_risk_level = '【风险等级】' in full_text
            has_risk_point = '【风险点】' in full_text
            has_feeding_advice = '【喂养建议】' in full_text
            
            if not has_feeding_advice:
                logger.warning("⚠️ AI回答缺少【喂养建议】部分，可能是max_tokens不足或被截断")
            
            formatted_text = format_ai_response(full_text)
            
            # 发送格式化后的文本（如果格式有变化）
            if formatted_text != full_text:
                yield f"data: {json.dumps({'formatted': formatted_text}, ensure_ascii=False)}\n\n"

        # 发送结束标记
        total_time = time.time() - start_time
        logger.info(f"✅ 流式响应完成，总耗时: {total_time:.3f}s")
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    except Exception as exc:
        logger.error("❌ 调用智谱 AI 失败：%s", exc)
        error_msg = json.dumps({"error": "AI 服务暂时不可用，请稍后再试。"}, ensure_ascii=False)
        yield f"data: {error_msg}\n\n"


@app.get("/")
async def root():
    """根路径，用于健康检查"""
    return {
        "status": "ok",
        "message": "宠物食品安全 AI 问答后端服务运行中",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

@app.post("/ask")
async def ask_ai(request: AskRequest):
    """处理前端发起的 AI 问答请求（流式响应）。"""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空，请提供宠物食品或健康相关的问题。")

    # 返回流式响应，使用 Server-Sent Events (SSE)
    # 传递宠物档案信息给流式响应函数
    return StreamingResponse(
        stream_zhipu_ai_response(question, request.pet_profile),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        }
    )


if __name__ == "__main__":
    # 当直接运行该文件时，启动 Uvicorn 服务器。
    # host=0.0.0.0 方便在局域网内访问，端口默认 3000，可通过环境变量 PORT 覆盖。
    import uvicorn

    port = int(os.getenv("PORT", "3000"))
    logger.info("✅ AI 问答接口已启动：http://localhost:%s", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

