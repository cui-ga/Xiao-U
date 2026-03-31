import gradio as gr
import sys
import os
import time
import traceback

# 将当前项目路径加入系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 助手配置
ASSISTANT_NAME = "小U"

# 全局医疗系统实例
medical_system = None
system_ready = False


def initialize_medical_system():
    """初始化医疗问答系统"""
    global medical_system, system_ready

    try:
        from QA_system.qa_coordinator import MedicalQAIntegratedSystem
        medical_system = MedicalQAIntegratedSystem()
        system_ready = True
        print("✅ 医疗系统初始化成功")
        return True
    except Exception as e:
        print(f"❌ 医疗系统初始化失败: {e}")
        return False


def get_medical_response(message):
    """获取医疗回复，只显示答案"""
    global medical_system, system_ready

    if not system_ready or medical_system is None:
        if not initialize_medical_system():
            return "⚠️ 系统正在初始化，请稍后再试..."

    try:
        # 生成会话ID
        session_id = f"user_{int(time.time())}"

        # 调用医疗系统
        result = medical_system.process_query(message, session_id=session_id)

        if result.get('success', False):
            # 只返回答案内容
            answer = result.get('answer', '抱歉，暂时没有找到相关信息。')
            return answer
        else:
            return "🤔 抱歉，我暂时无法回答这个问题。"

    except Exception as e:
        return f"⚠️ 系统处理出错: {str(e)[:100]}"


# 创建Gradio界面
with gr.Blocks(title=f"医疗助手 - {ASSISTANT_NAME}", fill_height=True) as demo:
    # 应用标题
    gr.HTML(f"""
    <div style="text-align: center; padding: 20px 20px 15px 20px;">
        <h1 style="margin: 0; font-size: 48px; font-weight: 700; color: #ffffff; letter-spacing: 1px;">{ASSISTANT_NAME}</h1>
        <p style="margin: 5px 0 0 0; color: #888; font-size: 16px; font-weight: 400;">智能医疗健康助手</p>
    </div>
    """)

    # 聊天区域 - 使用字典格式
    chatbot = gr.Chatbot(
        label="",
        height=300,
        show_label=False
    )

    # 输入区域
    with gr.Row():
        with gr.Column(scale=8):
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="尽管问... 例如：胃痛应该注意什么？",
                    scale=4,
                    container=False,
                    autofocus=True
                )
                submit_btn = gr.Button(
                    "发送",
                    variant="primary",
                    scale=1
                )

    # 底部信息
    gr.HTML(f"""
    <div style="text-align: center; margin-top: 15px; padding: 15px; color: #666; font-size: 12px;">
        <p style="margin: 0 0 5px 0;">© 2026 医疗助手 {ASSISTANT_NAME} | 本系统提供医疗信息参考，不能替代专业医生诊断</p>
    </div>
    """)


    # 事件处理 - 使用字典格式
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history

        # 获取医疗回复
        bot_response = get_medical_response(message)

        # 添加到历史 - 使用Gradio 6.10.0要求的字典格式
        # 注意：这里我们返回一个包含新消息的列表
        new_messages = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_response}
        ]

        return "", new_messages


    # 连接事件
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])

# 启动应用
if __name__ == "__main__":
    print("=" * 50)
    print(f"🏥 医疗助手 - {ASSISTANT_NAME}")
    print("=" * 50)
    print("🚀 系统启动中...")

    # 初始化医疗系统
    if initialize_medical_system():
        print("✅ 医疗系统准备就绪")
    else:
        print("⚠️ 医疗系统初始化失败，界面仍可运行")

    print(f"🌐 本地访问: http://localhost:7860")
    print("=" * 50)

    # 极简CSS
    custom_css = """
    /* 全局样式 */
    .gradio-container {
        margin: 0 auto;
        padding: 20px;
        max-width: 800px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0a0a0a;
        color: white;
    }

    /* 聊天框 */
    .chatbot {
        border: 1px solid #333;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
        height: 300px;
        overflow-y: auto;
        padding: 20px;
        margin-bottom: 15px;
    }

    /* 输入框 */
    .input-box textarea {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #333;
        background: rgba(255, 255, 255, 0.05);
        color: white;
        padding: 16px 20px;
        font-size: 16px;
    }

    .input-box textarea:focus {
        border-color: #6c5ce7;
        outline: none;
    }

    /* 发送按钮 */
    .send-button {
        background: #6c5ce7;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 16px;
        margin-left: 10px;
    }

    .send-button:hover {
        background: #5b4bd8;
    }

    /* 消息样式 */
    .user-message {
        background: rgba(108, 92, 231, 0.2);
        color: white;
        padding: 12px 18px;
        border-radius: 12px 12px 4px 12px;
        max-width: 80%;
        margin-left: auto;
        margin-bottom: 12px;
    }

    .bot-message {
        background: rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 12px 12px 12px 4px;
        max-width: 80%;
        margin-right: auto;
        margin-bottom: 12px;
    }
    """

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            css=custom_css
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️ 端口7860被占用，尝试使用7861端口...")
            demo.launch(
                server_name="0.0.0.0",
                server_port=7861,
                share=False,
                show_error=True,
                css=custom_css
            )
        else:
            raise