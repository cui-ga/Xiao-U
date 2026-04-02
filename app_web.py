import gradio as gr
import sys
import os
import time
import traceback
import uuid

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 助手配置
ASSISTANT_NAME = "小U"
PRIMARY_COLOR = "#6c5ce7"  
SECONDARY_COLOR = "#a29bfe"  
BACKGROUND_COLOR = "#0a0a0a"  
TEXT_COLOR = "#ffffff"  
BORDER_COLOR = "#333333"  

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


def get_medical_response(message, session_id):
    """获取医疗回复，支持多轮对话"""
    global medical_system, system_ready

    if not system_ready or medical_system is None:
        if not initialize_medical_system():
            return "⚠️ 系统正在初始化，请稍后再试...", session_id

    try:
        # 调用医疗系统，传递session_id
        result = medical_system.process_query(message, session_id=session_id)

        if result.get('success', False):
            # 只返回答案内容
            answer = result.get('answer', '抱歉，暂时没有找到相关信息。')
            return answer, session_id
        else:
            return "🤔 抱歉，我暂时无法回答这个问题。", session_id

    except Exception as e:
        return f"⚠️ 系统处理出错，请稍后再试。", session_id


# 创建Gradio界面
with gr.Blocks(title=f"医疗助手 - {ASSISTANT_NAME}", fill_height=True) as demo:
    # 使用Gradio的State来存储会话ID
    session_id_state = gr.State(value=f"web_session_{int(time.time())}")

    gr.HTML(f"""
    <div id="header-container" style="
        text-align: center; 
        padding: 20px; 
        background: {BACKGROUND_COLOR};
        border-bottom: 1px solid {BORDER_COLOR};
    ">
        <p style="
            margin: 0; 
            color: {PRIMARY_COLOR}; 
            font-size: 36px; 
            font-weight: 700; 
            letter-spacing: 1px;
        ">智能医疗健康助手 — 小U</p>
    </div>
    """)

    chatbot = gr.Chatbot(
        label="",
        show_label=False,
        elem_id="chatbot-container",
        value=[{"role": "assistant", "content": "您好！我是小U，您的智能医疗健康助手。有什么问题可以随时问我！"}]
    )

    # 输入区域
    with gr.Row(elem_id="input-row"):
        with gr.Column(scale=8):
            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="尽管问... 例如：糖尿病有什么症状？",
                    scale=4,
                    container=False,
                    autofocus=True,
                    elem_id="input-box"
                )
                submit_btn = gr.Button(
                    "发送",
                    variant="primary",
                    scale=1,
                    elem_id="submit-btn"
                )

    # 控制按钮
    with gr.Row(elem_id="control-row"):
        new_session_btn = gr.Button("🔄 新对话", variant="secondary", size="sm")
        clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="sm")

    gr.HTML(f"""
    <div id="footer-container" style="
        text-align: center; 
        padding: 12px; 
        color: #888888;  <!-- 调整为更易读的浅灰色 -->
        font-size: 12px;
        background: {BACKGROUND_COLOR};
        border-top: 1px solid {BORDER_COLOR};
    ">
        <p style="margin: 0;">© 2026 医疗助手 {ASSISTANT_NAME} | 本系统提供医疗信息参考，不能替代专业医生诊断</p>
    </div>
    """)


    # 事件处理
    def respond(message, chat_history, session_id):
        """处理用户消息，保持会话ID"""
        if not message.strip():
            return "", chat_history, session_id

        # 获取医疗回复，使用相同的session_id
        bot_response, session_id = get_medical_response(message, session_id)

        # 使用字典格式
        new_messages = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_response}
        ]

        return "", new_messages, session_id


    def start_new_session(chat_history, current_session_id):
        """开始新会话，生成新的session_id"""
        new_session_id = f"web_session_{int(time.time())}"
        print(f"🔄 开始新会话: {new_session_id}")
        return [], new_session_id


    def clear_chat(chat_history, session_id):
        """清空聊天记录，但保持会话ID"""
        print(f"🗑️ 清空会话 {session_id} 的聊天记录")
        return [], session_id


    # 连接事件
    msg.submit(
        respond,
        [msg, chatbot, session_id_state],
        [msg, chatbot, session_id_state]
    )

    submit_btn.click(
        respond,
        [msg, chatbot, session_id_state],
        [msg, chatbot, session_id_state]
    )

    new_session_btn.click(
        start_new_session,
        [chatbot, session_id_state],
        [chatbot, session_id_state]
    )

    clear_btn.click(
        clear_chat,
        [chatbot, session_id_state],
        [chatbot, session_id_state]
    )

# 启动应用
if __name__ == "__main__":
    print("=" * 50)
    print(f"🏥 医疗助手 - {ASSISTANT_NAME} (布局优化版)")
    print("=" * 50)
    print("🚀 系统启动中...")

    if initialize_medical_system():
        print("✅ 医疗系统准备就绪")
    else:
        print("⚠️ 医疗系统初始化失败，界面仍可运行")

    print(f"🌐 本地访问: http://localhost:7860")
    print("=" * 50)
    print("💡 多轮对话测试:")
    print("  1. 问: 糖尿病有什么症状？")
    print("  2. 问: 那怎么治疗？ (应该能理解'那'指糖尿病)")
    print("  3. 点击'新对话'按钮可以重置会话")
    print("=" * 50)

    # 【修正2】全屏CSS - 使用ID选择器替代顺序选择器，提高健壮性
    custom_css = f"""
    /* 重置所有边距和填充，确保全屏 */
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    /* 全局样式 - 全屏占满 */
    body, .gradio-container {{
        width: 100vw !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}

    /* 主容器 - 使用flex布局实现精确控制 */
    .gradio-container {{
        display: flex !important;
        flex-direction: column !important;
        max-width: 100% !important;
    }}

    /* 【关键修正】标题区域 - 通过ID精确选择，不再依赖DOM顺序 */
    #header-container {{
        flex: 0 0 auto !important;
        min-height: 80px !important;
        max-height: 80px !important;
    }}

    /* 聊天区域 - 自动填充剩余空间 */
    #chatbot-container {{
        flex: 1 1 auto !important;
        min-height: 0 !important; /* 重要：让flexbox正确计算 */
        border: none !important;
        border-radius: 0 !important;
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 20px !important;
        margin: 0 !important;
        overflow-y: auto !important;
    }}

    /* 控制按钮区域 */
    #control-row {{
        flex: 0 0 auto !important;
        min-height: 50px !important;
        max-height: 50px !important;
        padding: 0 20px 10px 20px !important;
        background: {BACKGROUND_COLOR} !important;
        display: flex !important;
        gap: 10px !important;
    }}

    /* 输入区域 - 固定高度 */
    #input-row {{
        flex: 0 0 auto !important;
        min-height: 80px !important;
        max-height: 80px !important;
        padding: 0 20px 10px 20px !important;
        background: {BACKGROUND_COLOR} !important;
        border-top: 1px solid {BORDER_COLOR} !important;
    }}

    /* 输入框 */
    #input-box textarea {{
        width: 100% !important;
        height: 52px !important;
        border-radius: 12px !important;
        border: 1px solid {BORDER_COLOR} !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: {TEXT_COLOR} !important;
        padding: 16px 20px !important;
        font-size: 16px !important;
        resize: none !important;
    }}

    #input-box textarea:focus {{
        border-color: {PRIMARY_COLOR} !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(108, 92, 231, 0.1) !important;
    }}

    #input-box textarea::placeholder {{
        color: #666 !important;
    }}

    /* 发送按钮 */
    #submit-btn {{
        background: {PRIMARY_COLOR} !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0 32px !important;
        height: 52px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
        margin-left: 10px !important;
    }}

    #submit-btn:hover {{
        background: #5b4bd8 !important;
        transform: translateY(-1px) !important;
    }}

    /* 控制按钮 */
    #control-row button {{
        background: rgba(255, 255, 255, 0.05) !important;
        color: {SECONDARY_COLOR} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        transition: all 0.3s !important;
    }}

    #control-row button:hover {{
        background: rgba(108, 92, 231, 0.1) !important;
        border-color: {PRIMARY_COLOR} !important;
    }}

    /* 消息样式 - 简洁美观 */
    .user-message {{
        background: rgba(108, 92, 231, 0.15) !important;
        color: {TEXT_COLOR} !important;
        padding: 12px 18px !important;
        border-radius: 12px 12px 4px 12px !important;
        max-width: 80% !important;
        margin-left: auto !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(108, 92, 231, 0.3) !important;
    }}

    .bot-message {{
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e0e0e0 !important;
        padding: 12px 18px !important;
        border-radius: 12px 12px 12px 4px !important;
        max-width: 80% !important;
        margin-right: auto !important;
        margin-bottom: 12px !important;
        border: 1px solid {BORDER_COLOR} !important;
    }}

    /* 聊天区域滚动条美化 */
    #chatbot-container::-webkit-scrollbar {{
        width: 6px;
    }}

    #chatbot-container::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }}

    #chatbot-container::-webkit-scrollbar-thumb {{
        background: {BORDER_COLOR};
        border-radius: 3px;
    }}

    #chatbot-container::-webkit-scrollbar-thumb:hover {{
        background: #444;
    }}

    /* 【关键修正】底部区域 - 通过ID精确选择 */
    #footer-container {{
        flex: 0 0 auto !important;
        min-height: 50px !important;
        max-height: 50px !important;
    }}

    /* 响应式调整 - 小屏幕优化 */
    @media (max-width: 768px) {{
        /* 标题区域缩小 */
        #header-container {{
            min-height: 60px !important;
            max-height: 60px !important;
        }}

        #header-container p {{
            font-size: 28px !important;
        }}

        /* 控制区域调整 */
        #control-row {{
            min-height: 45px !important;
            max-height: 45px !important;
            padding: 0 10px 8px 10px !important;
        }}

        /* 输入区域缩小 */
        #input-row {{
            min-height: 70px !important;
            max-height: 70px !important;
            padding: 0 10px 8px 10px !important;
        }}

        #input-box textarea {{
            height: 48px !important;
            padding: 12px 16px !important;
        }}

        #submit-btn {{
            height: 48px !important;
            padding: 0 20px !important;
        }}

        /* 底部区域缩小 */
        #footer-container {{
            min-height: 40px !important;
            max-height: 40px !important;
        }}

        #footer-container p {{
            font-size: 11px !important;
        }}
    }}

    /* 超小屏幕（手机）优化 */
    @media (max-height: 600px) {{
        /* 标题区域进一步缩小 */
        #header-container {{
            min-height: 50px !important;
            max-height: 50px !important;
        }}

        #header-container p {{
            font-size: 24px !important;
        }}

        /* 聊天区域减少内边距 */
        #chatbot-container {{
            padding: 10px !important;
        }}

        .user-message, .bot-message {{
            padding: 10px 14px !important;
            margin-bottom: 8px !important;
        }}

        /* 底部区域进一步缩小 */
        #footer-container {{
            min-height: 35px !important;
            max-height: 35px !important;
        }}

        #footer-container p {{
            font-size: 10px !important;
        }}
    }}
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
