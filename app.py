import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

if "--device" in sys.argv:
    try:
        device_idx = sys.argv.index("--device")
        device_value = sys.argv[device_idx + 1]
        os.environ["DOCLING_DEVICE"] = device_value
        print(f"[INFO] DOCLING_DEVICE set to: {device_value}")
    except (IndexError, ValueError):
        print("[WARNING] --device requires a value (cpu or cuda)")

os.environ["GRADIO_LANGUAGE"] = "en"  

import uuid
import time
import shutil
import threading
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
from fastapi import FastAPI

def _noop(self, app: FastAPI):
    pass

gr.blocks.Blocks._add_health_routes = _noop

from rebuttal_service import (
    rebuttal_service,
    ProcessStatus,
    SessionState,
    QuestionState,
    init_llm_client,
    LogCollector,
)


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_CURRENT_DIR, "gradio_uploads")
os.makedirs(SAVE_DIR, exist_ok=True)

def read_gradio_file(file_obj) -> Tuple[Optional[str], Optional[Any]]:
    if file_obj is None:
        return None, None

    if isinstance(file_obj, str):
        return "path", file_obj
    if isinstance(file_obj, dict) and "data" in file_obj:
        return "bytes", file_obj["data"]
    if hasattr(file_obj, "read"):
        return "fileobj", file_obj.read()

    raise ValueError(f"Unknown gr.File object format: {type(file_obj)}")


def save_uploaded_files(pdf_file, review_file, session_id: str) -> Tuple[str, str, str]:
    session_dir = os.path.join(SAVE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    pdf_save_path = os.path.join(session_dir, "paper.pdf")
    review_save_path = os.path.join(session_dir, "review.txt")
    
    pdf_type, pdf_data = read_gradio_file(pdf_file)
    if pdf_type is None:
        raise ValueError("PDF file upload failed or incorrect format")
    if pdf_type == "path":
        shutil.copy(pdf_data, pdf_save_path)
    elif pdf_type in ("bytes", "fileobj"):
        with open(pdf_save_path, "wb") as f:
            f.write(pdf_data if isinstance(pdf_data, bytes) else pdf_data)
    
    rev_type, rev_data = read_gradio_file(review_file)
    if rev_type is None:
        raise ValueError("Review file upload failed or incorrect format")
    review_text = ""
    

    def decode_with_fallback(data: bytes) -> str:
        """Try multiple encodings to decode bytes, with UTF-8 as primary."""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        for enc in encodings:
            try:
                return data.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue

        return data.decode('utf-8', errors='replace')
    
    if rev_type == "path":
        with open(rev_data, "rb") as f:
            raw_bytes = f.read()
        review_text = decode_with_fallback(raw_bytes)
    elif rev_type in ("bytes", "fileobj"):
        if isinstance(rev_data, bytes):
            review_text = decode_with_fallback(rev_data)
        else:
            review_text = decode_with_fallback(rev_data)
    
    with open(review_save_path, "w", encoding="utf-8") as f:
        f.write(review_text)
    
    return pdf_save_path, review_save_path, review_text


processing_threads: Dict[str, threading.Thread] = {}

# Provider configurations
PROVIDER_CONFIGS = {
    "OpenRouter": {
        "provider_key": "openrouter",
        "env_var": "OPENROUTER_API_KEY",
        "label": "OpenRouter API Key",
        "placeholder": "sk-or-v1-...",
    },
    "Qwen (DashScope)": {
        "provider_key": "qwen",
        "env_var": "QWEN_API_KEY",
        "label": "Qwen API Key",
        "placeholder": "sk-...",
    },
    "DeepSeek": {
        "provider_key": "deepseek",
        "env_var": "DEEPSEEK_API_KEY",
        "label": "DeepSeek API Key",
        "placeholder": "sk-...",
    },
    "OpenAI": {
        "provider_key": "openai",
        "env_var": "OPENAI_API_KEY",
        "label": "OpenAI API Key",
        "placeholder": "sk-...",
    },
    "Gemini": {
        "provider_key": "gemini",
        "env_var": "GEMINI_API_KEY",
        "label": "Gemini API Key",
        "placeholder": "AIza...",
    },
    "ZhiPu (GLM)": {
        "provider_key": "zhipu",
        "env_var": "ZHIPUAI_API_KEY",
        "label": "ZhiPu API Key",
        "placeholder": "...",
    },
}

# Model choices per provider
MODEL_CHOICES_BY_PROVIDER = {
    "OpenRouter": {
        "Gemini 3 Flash": "google/gemini-3-flash-preview",
        "Grok 4.1 Fast": "x-ai/grok-4.1-fast",
        "GPT-5 Mini": "openai/gpt-5-mini",
        "DeepSeek V3.2": "deepseek/deepseek-chat-v3.2",
        "Other models": "custom",
    },
    "Qwen (DashScope)": {
        "Qwen-Turbo": "qwen-turbo",
        "Qwen-Plus": "qwen-plus",
        "Qwen-Max": "qwen-max",
        "Other models": "custom",
    },
    "DeepSeek": {
        "DeepSeek Chat": "deepseek-chat",
        "DeepSeek Reasoner": "deepseek-reasoner",
        "Other models": "custom",
    },
    "OpenAI": {
        "GPT-5.2": "gpt-5.2",
        "GPT-5 Mini": "gpt-5-mini",
        "Other models": "custom",
    },
    "Gemini": {
        "Gemini-3-Pro": "gemini-3-pro-preview",
        "Gemini-3-Flash": "models/gemini-3-flash-preview",
        "Other models": "custom",
    },
    "ZhiPu (GLM)": {
        "GLM-4.7": "glm-4.7",
        "Other models": "custom",
    },
}




def get_api_key_for_provider(provider: str) -> str:
    """Get API key from environment for specified provider"""
    config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["OpenRouter"])
    return os.environ.get(config["env_var"], "")


def get_default_model_for_provider(provider: str) -> str:
    """Get default model for specified provider"""
    models = MODEL_CHOICES_BY_PROVIDER.get(provider, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
    # Return first model (excluding "Other models")
    for name, value in models.items():
        if name != "Other models":
            return name
    return list(models.keys())[0]


def start_analysis(pdf_file, review_file, provider_choice, api_key, model_choice, custom_model):
    if not pdf_file or not review_file:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ Please upload paper PDF and review file!",
            gr.Timer(active=False),  
        )
    
    if not api_key or not api_key.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ Please enter API Key!",
            gr.Timer(active=False),
        )
    
    # Get provider key from config
    provider_config = PROVIDER_CONFIGS.get(provider_choice, PROVIDER_CONFIGS["OpenRouter"])
    provider_key = provider_config["provider_key"]
    
    # Get model choices for this provider
    model_choices = MODEL_CHOICES_BY_PROVIDER.get(provider_choice, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
    
    if model_choice == "Other models":
        if not custom_model or not custom_model.strip():
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                None,
                "⚠️ Please enter custom model name!",
                gr.Timer(active=False),
            )
        selected_model = custom_model.strip()
    else:
        selected_model = model_choices.get(model_choice, list(model_choices.values())[0])
    
    session_id = str(uuid.uuid4())[:8]
    
    try:
        init_llm_client(api_key=api_key.strip(), provider=provider_key, model=selected_model)
        pdf_path, review_path, _ = save_uploaded_files(pdf_file, review_file, session_id)
        session = rebuttal_service.create_session(session_id, pdf_path, review_path)
        
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            {"session_id": session_id, "current_idx": 0},
            "📤 Files uploaded successfully, initializing analysis...",
            gr.Timer(active=True),  
        )
        
    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            f"❌ Processing failed: {str(e)}",
            gr.Timer(active=False),
        )


def run_initial_analysis(session_state):
    if not session_state:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            session_state,
            "❌ Session state lost",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.Timer(active=False), 
        )
    
    session_id = session_state.get("session_id")
    
    try:
        session = rebuttal_service.run_initial_analysis(session_id)
        rebuttal_service.process_all_questions_parallel(session_id, max_workers=3)
        session = rebuttal_service.get_session(session_id)
        
        session_state["current_idx"] = 0
        q_state = session.questions[0]
        
        history_text = format_feedback_history(q_state.feedback_history)
        strategy_content = q_state.agent7_output or ""
        
        # Determine button text based on whether this is the last question
        is_last_question = len(session.questions) == 1
        btn_text = "📝 Generate Final Rebuttal" if is_last_question else "✅ Satisfied, Next Question"
        
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            session_state,
            "",
            f"### Question 1 / {len(session.questions)}",
            q_state.question_text,
            strategy_content,
            strategy_content,
            "",
            f"📝 Revisions have been revised {q_state.revision_count} times",
            gr.update(interactive=True),
            history_text,
            gr.Timer(active=False),
            gr.update(value=btn_text),
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            session_state,
            f"❌ Analysis failed : {str(e)}",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.Timer(active=False),
            gr.update(),
        )


def regenerate_strategy(feedback_text, session_state):
    if not session_state:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            session_state,
        )
    
    if not feedback_text or not feedback_text.strip():
        return (
            gr.update(),
            gr.update(),
            "⚠️ Please enter feedback ",
            gr.update(),
            session_state,
        )
    
    session_id = session_state.get("session_id")
    current_idx = session_state.get("current_idx", 0)
    
    try:
        q_state = rebuttal_service.revise_with_feedback(
            session_id, 
            current_idx, 
            feedback_text.strip()
        )
        
        history_text = format_feedback_history(q_state.feedback_history)
        strategy_content = q_state.agent7_output or ""
        
        return (
            strategy_content,
            strategy_content,
            "",
            f"📝 Revisions have been revised {q_state.revision_count} times ✓ Latest revision applied",
            history_text,
            session_state,
        )
        
    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f"❌ Revision failed: {str(e)}",
            gr.update(),
            session_state,
        )


def format_feedback_history(history: list) -> str:
    if not history:
        return "*No revisions yet*"
    
    lines = []
    for i, record in enumerate(history, 1):
        feedback = record.get("feedback", "")
        if len(feedback) > 100:
            feedback = feedback[:100] + "..."
        lines.append(f"**#{i}** {feedback}")
    
    return "\n\n".join(lines)


def generate_strategy_summary(session) -> str:
    lines = []
    lines.append(" This document contains all questions' rebuttal strategies and To-Do List\n")
    lines.append("=" * 60 + "\n")
    
    for q in session.questions:
        lines.append(f"## Q{q.question_id}: {q.question_text[:100]}{'...' if len(q.question_text) > 100 else ''}")
        lines.append("")
        lines.append("### Rebuttal strategy & To-Do List")
        lines.append("")
        lines.append(q.agent7_output if q.agent7_output else "**Not generated**")
        lines.append("")
        if q.revision_count > 0:
            lines.append(f"> 📝 Revisions have been revised {q.revision_count} times")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)


def skip_question(session_state):
    if not session_state:
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(),
        )
    
    session_id = session_state.get("session_id")
    current_idx = session_state.get("current_idx", 0)
    
    try:
        session = rebuttal_service.get_session(session_id)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        rebuttal_service.mark_question_satisfied(session_id, current_idx)
        
        next_idx = current_idx + 1
        
        if next_idx < len(session.questions):
            q_state = session.questions[next_idx]
            session_state["current_idx"] = next_idx
            
            history_text = format_feedback_history(q_state.feedback_history)
            
            strategy_content = q_state.agent7_output or ""
            
            # Determine button text based on whether this is the last question
            is_last_question = (next_idx + 1) == len(session.questions)
            btn_text = "📝 Generate Final Rebuttal" if is_last_question else "✅ Satisfied, Next Question"
            
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                session_state,
                f"### Question {next_idx + 1} / {len(session.questions)}",
                q_state.question_text,
                strategy_content,
                strategy_content,
                "",
                f"📝 Revisions have been revised {q_state.revision_count} times",
                gr.update(interactive=True),
                history_text,
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(value=btn_text),
            )
        else:
            strategy_summary = generate_strategy_summary(session)
            final_text = rebuttal_service.generate_final_rebuttal(session_id)
            
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                session_state,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                strategy_summary, strategy_summary, final_text, final_text,
                gr.update(),
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), f"❌ Processing failed: {str(e)}",
            gr.update(),
        )


def confirm_and_next(strategy_text, session_state):
    if not session_state:
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(),
        )
    
    session_id = session_state.get("session_id")
    current_idx = session_state.get("current_idx", 0)
    
    try:
        session = rebuttal_service.get_session(session_id)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.questions[current_idx].agent7_output = strategy_text
        rebuttal_service.mark_question_satisfied(session_id, current_idx)
        
        next_idx = current_idx + 1
        
        if next_idx < len(session.questions):
            q_state = session.questions[next_idx]
            session_state["current_idx"] = next_idx
            
            history_text = format_feedback_history(q_state.feedback_history)
            
            strategy_content = q_state.agent7_output or ""
            
            # Determine button text based on whether this is the last question
            is_last_question = (next_idx + 1) == len(session.questions)
            btn_text = "📝 Generate Final Rebuttal" if is_last_question else "✅ Satisfied, Next Question"
            
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                session_state,
                f"### Question {next_idx + 1} / {len(session.questions)}",
                q_state.question_text,
                strategy_content,
                strategy_content,
                "",
                f"📝 Revisions have been revised {q_state.revision_count} times",
                gr.update(interactive=True),
                history_text,
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(value=btn_text),
            )
        else:
            strategy_summary = generate_strategy_summary(session)
            final_text = rebuttal_service.generate_final_rebuttal(session_id)
            
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                session_state,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                strategy_summary, strategy_summary, final_text, final_text,
                gr.update(),
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), f"❌ Processing failed: {str(e)}",
            gr.update(),
        )


def restart_session():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        None,
        "",
        None, None,
    )


def get_active_sessions_choices():
    """Get choices for session dropdown"""
    sessions = rebuttal_service.list_active_sessions()
    if not sessions:
        return []
    return [(s["display_text"], s["session_id"]) for s in sessions]


def refresh_session_list():
    """Refresh the session dropdown choices"""
    choices = get_active_sessions_choices()
    if not choices:
        return gr.update(choices=[], value=None), "📭 No active sessions found"
    return gr.update(choices=choices, value=choices[0][1]), f"🔄 Found {len(choices)} active session(s)"


def resume_session(session_id_to_resume, provider_choice, api_key):
    """Resume an existing session after page refresh"""
    if not session_id_to_resume:
        return (
            gr.update(),  # upload_col
            gr.update(),  # loading_col
            gr.update(),  # interact_col
            gr.update(),  # result_col
            None,         # session_state
            "⚠️ Please select a session to resume!",  # upload_status
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(),  # confirm_btn
        )
    
    if not api_key or not api_key.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ Please enter API Key before resuming!",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(),
        )
    
    try:
        # Initialize LLM client with provided credentials
        provider_config = PROVIDER_CONFIGS.get(provider_choice, PROVIDER_CONFIGS["OpenRouter"])
        provider_key = provider_config["provider_key"]
        model_choices = MODEL_CHOICES_BY_PROVIDER.get(provider_choice, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
        default_model = list(model_choices.values())[0]
        init_llm_client(api_key=api_key.strip(), provider=provider_key, model=default_model)
        
        session = rebuttal_service.get_session(session_id_to_resume)
        if not session:
            session = rebuttal_service.restore_session_from_disk(session_id_to_resume)
        if not session:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                None,
                f"❌ Session {session_id_to_resume} not found!",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                gr.update(),
            )
        
        # Check if session has questions processed
        if not session.questions:
            return (
                gr.update(),
                gr.update(visible=True),  # Show loading page
                gr.update(),
                gr.update(),
                {"session_id": session_id_to_resume, "current_idx": 0},
                "📤 Session found but still processing. Please wait...",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                gr.update(),
            )
        
        # Find first unprocessed or unsatisfied question
        resume_idx = 0
        for i, q in enumerate(session.questions):
            if not q.is_satisfied and q.agent7_output:
                resume_idx = i
                break
            elif q.is_satisfied:
                resume_idx = i + 1
        
        # If all questions are satisfied, go to result page
        if resume_idx >= len(session.questions):
            strategy_summary = generate_strategy_summary(session)
            final_text = session.final_rebuttal or rebuttal_service.generate_final_rebuttal(session_id_to_resume)
            
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),  # Show result page
                {"session_id": session_id_to_resume, "current_idx": resume_idx - 1},
                "",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                gr.update(),
            )
        
        # Resume to the question review page
        q_state = session.questions[resume_idx]
        history_text = format_feedback_history(q_state.feedback_history)
        strategy_content = q_state.agent7_output or ""
        
        is_last_question = (resume_idx + 1) == len(session.questions)
        btn_text = "📝 Generate Final Rebuttal" if is_last_question else "✅ Satisfied, Next Question"
        
        return (
            gr.update(visible=False),  # upload_col
            gr.update(visible=False),  # loading_col
            gr.update(visible=True),   # interact_col
            gr.update(visible=False),  # result_col
            {"session_id": session_id_to_resume, "current_idx": resume_idx},  # session_state
            "",  # upload_status
            f"### Question {resume_idx + 1} / {len(session.questions)} (Resumed)",  # progress_info
            q_state.question_text,  # question_display
            strategy_content,  # strategy_preview
            strategy_content,  # strategy_edit
            "",  # feedback_input
            f"📝 Revisions have been revised {q_state.revision_count} times",  # revision_info
            gr.update(interactive=True),  # regenerate_btn
            history_text,  # feedback_history_display
            gr.update(value=btn_text),  # confirm_btn
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            f"❌ Failed to resume session: {str(e)}",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(),
        )


def poll_logs(session_state):
    """Poll logs for real-time updates on the loading page"""
    if not session_state:
        return gr.update(), session_state
    
    session_id = session_state.get("session_id")
    if not session_id:
        return gr.update(), session_state
    
    session = rebuttal_service.get_session(session_id)
    if not session or not session.log_collector:
        return gr.update(), session_state
    
    logs = session.log_collector.get_recent(30)
    if not logs:
        return gr.update(), session_state
    
    prev_logs = session_state.get("_prev_logs", "")
    if logs == prev_logs:
        return gr.update(), session_state
    
    session_state["_prev_logs"] = logs
    return logs, session_state



# CSS for the application
APP_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
    
    /* Global fonts: Georgia for English, Noto Sans SC for Chinese */
    * {
        font-family: Georgia, 'Noto Sans SC', 'PingFang SC', 'Hiragino Sans GB', sans-serif !important;
    }
    .prose, .prose * {
        font-family: Georgia, 'Noto Sans SC', 'PingFang SC', 'Hiragino Sans GB', sans-serif !important;
    }
    /* Code blocks keep monospace font */
    code, pre, .code, pre *, code * {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    }
    .strategy-preview {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        line-height: 1.8;
        max-height: 600px;
        overflow-y: auto;
    }
    .strategy-preview h3 {
        color: #1e40af;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 8px;
        margin-top: 20px;
    }
    .strategy-preview h4 {
        color: #7c3aed;
        margin-top: 16px;
    }
    .strategy-preview strong {
        color: #1e293b;
        border-radius: 4px; /* Optional: adds subtle highlight connection */
    }
    .strategy-preview table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
    }
    .strategy-preview th, .strategy-preview td {
        border: 1px solid #e2e8f0;
        padding: 8px 12px;
        text-align: left;
    }
    .strategy-preview th {
        background: #f1f5f9;
    }
    .strategy-edit textarea {
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .question-box {
        background: linear-gradient(135deg, #fef3c7, #fef9c3);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
    }
    .feedback-box textarea {
        border: 2px solid #4CAF50;
    }
    #log-display {
        background: #f8fafc;
        color: #334155;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
    }
    /* Download tip animation */
    @keyframes pulse-glow {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.7;
            transform: scale(1.02);
        }
    }
    @keyframes arrow-bounce {
        0%, 100% { transform: translateX(0); }
        50% { transform: translateX(5px); }
    }
    .download-tip {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 12px;
        animation: pulse-glow 2s ease-in-out infinite;
    }
    .download-tip em {
        font-style: normal;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* Important warning notice - single layer only */
    .important-warning {
        background: linear-gradient(135deg, #fef2cd, #fff3cd) !important;
        border: 2px solid #ff9800 !important;
        border-left: 6px solid #ff5722 !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        margin: 16px 0 !important;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.25) !important;
    }
    .important-warning * {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .important-warning p {
        margin: 4px 0 !important;
        color: #5d4037 !important;
        font-weight: 500 !important;
    }
    /* Bright download buttons */
    #download-strategy-btn, #download-rebuttal-btn {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 16px 24px !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    #download-strategy-btn:hover, #download-rebuttal-btn:hover {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.6) !important;
        transform: translateY(-2px) !important;
    }
"""

with gr.Blocks(title="AI Rebuttal Assistant") as demo:
    
    session_state = gr.State(None)
    
    gr.Markdown(
        """
        # AI Rebuttal Assistant
        
        **Workflow usage process:**
        - **Upload** - Upload your paper PDF and the review file (Supports .txt or .md format, please input the original document containing all reviewers' comments, including reviewers' IDs ; there is no need to split them manually)
        - **Analysis** - The system will automatically analyze your paper and extract questions from the review
        - **Review Strategy** - For each question, view the AI-generated rebuttal strategy and to-do list and referenced response snippets
        - **Refinement** - Enter your feedback and click "Regenerate" to refine the strategy
        - **Generate Rebuttal** - After all questions are processed, generate the final rebuttal document
        """
    )
    
    with gr.Column(visible=True) as upload_col:
        gr.Markdown("## 📤 Configure & Upload Files")
        
        with gr.Group():
            gr.Markdown("### 🔑 API Configuration")
            
            # Provider selection
            provider_choice = gr.Dropdown(
                label="LLM Provider",
                choices=list(PROVIDER_CONFIGS.keys()),
                value="OpenRouter",
                info="Select your LLM provider",
            )
            
            # Pre-fill API key from environment variable based on provider
            default_provider = "OpenRouter"
            env_api_key = get_api_key_for_provider(default_provider)
            api_key_input = gr.Textbox(
                label=PROVIDER_CONFIGS[default_provider]["label"],
                placeholder=f"Please enter your API Key ({PROVIDER_CONFIGS[default_provider]['placeholder']})",
                value=env_api_key,
                type="password",
                info="Your API key will not be stored, only used for this session." + (" (Loaded from .env)" if env_api_key else "")
            )
            
            def on_provider_change(provider):
                """Update API key field and model choices when provider changes"""
                config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["OpenRouter"])
                env_key = get_api_key_for_provider(provider)
                model_choices = MODEL_CHOICES_BY_PROVIDER.get(provider, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
                default_model = get_default_model_for_provider(provider)
                
                return (
                    gr.update(
                        label=config["label"],
                        placeholder=f"Please enter your API Key ({config['placeholder']})",
                        value=env_key,
                        info="Your API key will not be stored, only used for this session." + (" (Loaded from .env)" if env_key else "")
                    ),
                    gr.update(
                        choices=list(model_choices.keys()),
                        value=default_model,
                    ),
                )
        
        gr.Markdown("---")
        
        with gr.Group():
            gr.Markdown("### 🤖 Model Selection")
            with gr.Row():
                model_choice = gr.Dropdown(
                    label="Select Model",
                    choices=list(MODEL_CHOICES_BY_PROVIDER["OpenRouter"].keys()),
                    value="Gemini 3 Flash",
                    info="Choose the LLM model to use",
                    scale=2,
                )
                custom_model_input = gr.Textbox(
                    label="Custom Model Name",
                    placeholder="Enter model name",
                    visible=False,
                    scale=3,
                )
            
            def toggle_custom_model(choice):
                return gr.update(visible=(choice == "Other models"))
            
            model_choice.change(
                fn=toggle_custom_model,
                inputs=[model_choice],
                outputs=[custom_model_input],
            )
            
            # Connect provider change to update API key and model choices
            provider_choice.change(
                fn=on_provider_change,
                inputs=[provider_choice],
                outputs=[api_key_input, model_choice],
            )
        
        gr.Markdown("---")
        
        gr.Markdown("### 📄 Upload Files")
        with gr.Row():
            pdf_input = gr.File(
                label="📄 Paper PDF", 
                file_types=[".pdf"],
                file_count="single",
            )
            review_input = gr.File(
                label="📝 Review File(.md / .txt)", 
                file_types=[".md", ".txt"],
                file_count="single",
            )
        
        upload_status = gr.Markdown("")
        
        start_btn = gr.Button(
            "🚀 Submit & Start Analysis", 
            variant="primary",
            size="lg",
        )
        
        gr.Markdown("---")
        
        with gr.Group():
            gr.Markdown("### 🔄 Resume Previous Session")
            gr.Markdown(
                "*If you refreshed the page during processing, you can resume your session here. "
                "Make sure to enter your API Key above first.*"
            )
            with gr.Row():
                resume_session_dropdown = gr.Dropdown(
                    label="Select Session to Resume",
                    choices=[],
                    value=None,
                    interactive=True,
                    scale=3,
                )
                refresh_sessions_btn = gr.Button(
                    "🔄 Refresh List",
                    variant="secondary",
                    size="sm",
                    scale=1,
                )
            resume_status = gr.Markdown("")
            resume_btn = gr.Button(
                "▶️ Resume Selected Session",
                variant="secondary",
                size="lg",
            )
    
    with gr.Column(visible=False) as loading_col:
        gr.Markdown("## ⏳ Analyzing...")
        loading_status = gr.Markdown("Initializing...")
        
        gr.Markdown(
            """
            > 📊 **Analysis Process:**
            > 1. Convert PDF to Markdown 
            > 2. AI reads and summarizes the paper 
            > 3. AI extracts questions from the review 
            > 4. Process all questions in parallel 
            > 5. Present results for your review 
            
            🚀 **All questions will be processed in parallel**, so you can quickly review and refine each one after completion!

            After all questions are processed, you can generate the referenced final rebuttal document.
            
            This may take about 15 minutes (use gpu) or up to 1 hour (CPU), please be patient...
            """
        )
        
        gr.Markdown("### 📋 Live Logs")
        log_display = gr.Textbox(
            value="Waiting to start...",
            label="",
            lines=10,
            max_lines=15,
            interactive=False,
            elem_id="log-display",
        )

        log_timer = gr.Timer(value=1.5, active=False)
    
    with gr.Column(visible=False) as interact_col:
        with gr.Row():
            progress_info = gr.Markdown("### Question 1 / N")
            processing_status = gr.Markdown("", elem_id="processing-status")
        
        gr.Markdown(
            """
            > 📘 **Quick Reference:**
            > - **Strategy** — High-level approach and key arguments to address this reviewer question
            > - **To-Do List** — Concrete action items (experiments, analysis, writing) to implement the strategy
            > - **Response Draft** — Snippets you can refer to when writing your rebuttal
            """
        )
        
        with gr.Group():
            gr.Markdown("#### 🔍 Reviewer's Question")
            question_display = gr.Markdown(elem_classes=["question-box"])
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("#### 💡 Rebuttal Strategy & To-Do List Rebuttal")
                
                with gr.Tabs():
                    with gr.TabItem("📖 Preview"):
                        gr.Markdown("*Rendered strategy content below:*")
                        strategy_preview = gr.Markdown(elem_classes=["strategy-preview"])
                    
                    with gr.TabItem("✏️ Edit"):
                        gr.Markdown("*Edit raw Markdown, switch back to Preview to see results:*")
                        strategy_edit = gr.Textbox(
                            label="",
                            lines=20,
                            max_lines=40,
                            elem_classes=["strategy-edit"],
                        )
                
                revision_info = gr.Markdown("📝 Revisions have been modified 0 times")
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📜 Revision History")
                feedback_history_display = gr.Markdown(
                    "*No revisions yet*",
                    elem_id="feedback-history",
                )
        
        gr.Markdown("---")
        gr.Markdown("#### 💬 Human Feedback")
        gr.Markdown("*Enter your feedback, and the AI will adjust the strategy accordingly. You can revise multiple times until satisfied.*")
        
        with gr.Row():
            feedback_input = gr.Textbox(
                label="Feedback", 
                placeholder="For example:\n• Please add more discussion on experimental data\n• This strategy is too general, needs more specific suggestions\n• Please include a comparison with baseline methods",
                lines=4,
                scale=4,
                elem_classes=["feedback-box"],
            )
            with gr.Column(scale=1):
                regenerate_btn = gr.Button(
                    "🔄 Regenerate", 
                    variant="secondary",
                    size="lg",
                )
                gr.Markdown("*AI will revise strategy\nbased on your feedback*", elem_id="regen-hint")
        
        gr.Markdown("---")
        
        with gr.Row():
            skip_btn = gr.Button(
                "⏭️ Skip This Question",
                variant="secondary",
                size="lg",
            )
            confirm_btn = gr.Button(
                "✅ Satisfied, Next Question", 
                variant="primary",
                size="lg",
            )
    
    with gr.Column(visible=False) as result_col:
        gr.Markdown("## 🎉 Complete!")
        gr.Markdown(
            """
            All questions have been processed. Here are the generated results:
            
            This page contains **two modules**:
            1. **Strategy Summary** - Contains rebuttal strategies, To-Do Lists, and draft response snippets for all questions.
            2. **Final Reference Rebuttal** - A complete rebuttal document for your reference.
            """
        )
        
        gr.Markdown(
            """
            ⚠️ **IMPORTANT NOTICE:** The Final Reference Rebuttal contains **LLM-estimated numerical values marked with asterisks (*)**. 
            These estimated values are placeholders and **MUST be replaced with actual experimental results**.
            Please carefully review and verify all data.
            """,
            elem_classes=["important-warning"]
        )
        
        with gr.Tabs():
            with gr.TabItem("📝 Strategy Summary"):
                gr.Markdown("*Contains rebuttal strategies, To-Do Lists, and draft response snippets for all questions.*")
                with gr.Tabs():
                    with gr.TabItem("📖 Preview"):
                        strategy_summary_preview = gr.Markdown(elem_classes=["strategy-preview"])
                    with gr.TabItem("✏️ Raw Text"):
                        strategy_summary_output = gr.Textbox(
                            label="Rebuttal Strategy & To-Do List", 
                            lines=20,
                            max_lines=40,
                        )
            
            with gr.TabItem("📄 Final Reference Rebuttal"):
                gr.Markdown(
                    """
                    *The complete reference rebuttal document.*
                    
                    > ⚠️ **Note:** Numerical values marked with **asterisks (*)** are LLM-estimated placeholders.
                    > You **MUST supplement these with actual experimental data** .
                    """
                )
                with gr.Tabs():
                    with gr.TabItem("📖 Preview"):
                        final_preview = gr.Markdown(elem_classes=["strategy-preview"])
                    with gr.TabItem("✏️ Raw Text"):
                        final_output = gr.Textbox(
                            label="Final Reference Rebuttal", 
                            lines=20,
                            max_lines=40,
                        )
        
        gr.Markdown("---")
        gr.Markdown("### 📥 Download Files")
        
        with gr.Row():
            download_strategy_btn = gr.Button(
                "📥 Download Strategy Summary", 
                variant="primary",
                size="lg",
                elem_id="download-strategy-btn",
            )
            download_rebuttal_btn = gr.Button(
                "📥 Download Reference Rebuttal", 
                variant="primary",
                size="lg",
                elem_id="download-rebuttal-btn",
            )
            restart_btn = gr.Button(
                "🔄 Start Over", 
                variant="secondary",
                size="lg",
            )
        
        download_strategy_file = gr.File(label="Strategy File", visible=False)
        download_rebuttal_file = gr.File(label="Rebuttal File", visible=False)
        
        gr.Markdown(
            "💡 **Tip:** After clicking the download button, click the **file size link on the right** ➡️ of the file component to start download.",
            elem_classes=["download-tip"]
        )
    
    start_btn.click(
        fn=start_analysis,
        inputs=[pdf_input, review_input, provider_choice, api_key_input, model_choice, custom_model_input],
        outputs=[
            upload_col, loading_col, interact_col, result_col,
            session_state, upload_status, log_timer,
        ],
    ).then(
        fn=run_initial_analysis,
        inputs=[session_state],
        outputs=[
            upload_col, loading_col, interact_col, result_col,
            session_state, loading_status,
            progress_info, question_display, strategy_preview, strategy_edit, feedback_input,
            revision_info, regenerate_btn, feedback_history_display, log_timer,
            confirm_btn,
        ],
    )

    refresh_sessions_btn.click(
        fn=refresh_session_list,
        inputs=[],
        outputs=[resume_session_dropdown, resume_status],
    )
    
    resume_btn.click(
        fn=resume_session,
        inputs=[resume_session_dropdown, provider_choice, api_key_input],
        outputs=[
            upload_col, loading_col, interact_col, result_col,
            session_state, upload_status,
            progress_info, question_display, strategy_preview, strategy_edit, feedback_input,
            revision_info, regenerate_btn, feedback_history_display,
            confirm_btn,
        ],
    )

    log_timer.tick(
        fn=poll_logs,
        inputs=[session_state],
        outputs=[log_display, session_state],
    )
    
    regenerate_btn.click(
        fn=regenerate_strategy,
        inputs=[feedback_input, session_state],
        outputs=[strategy_preview, strategy_edit, feedback_input, revision_info, feedback_history_display, session_state],
    )
    
    def sync_preview(text):
        return text
    
    strategy_edit.blur(
        fn=sync_preview,
        inputs=[strategy_edit],
        outputs=[strategy_preview],
    )
    
    confirm_btn.click(
        fn=confirm_and_next,
        inputs=[strategy_edit, session_state],
        outputs=[
            interact_col, result_col, session_state,
            progress_info, question_display, strategy_preview, strategy_edit, feedback_input,
            revision_info, regenerate_btn, feedback_history_display,
            strategy_summary_preview, strategy_summary_output, final_preview, final_output,
            confirm_btn,
        ],
    )
    
    skip_btn.click(
        fn=skip_question,
        inputs=[session_state],
        outputs=[
            interact_col, result_col, session_state,
            progress_info, question_display, strategy_preview, strategy_edit, feedback_input,
            revision_info, regenerate_btn, feedback_history_display,
            strategy_summary_preview, strategy_summary_output, final_preview, final_output,
            confirm_btn,
        ],
    )
    
    restart_btn.click(
        fn=restart_session,
        inputs=[],
        outputs=[
            upload_col, loading_col, interact_col, result_col,
            session_state, upload_status,
            pdf_input, review_input,
        ],
    )
    
    def download_strategy(strategy_text):
        if not strategy_text:
            return gr.update()
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='_strategy.md', delete=False, encoding='utf-8') as f:
            f.write("# Rebuttal Strategy & To-Do List \n\n")
            f.write(strategy_text)
            return gr.update(value=f.name, visible=True)
    
    download_strategy_btn.click(
        fn=download_strategy,
        inputs=[strategy_summary_output],
        outputs=[download_strategy_file],
    )
    
    def download_rebuttal(final_text):
        if not final_text:
            return gr.update()
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='_rebuttal.md', delete=False, encoding='utf-8') as f:
            f.write("# Final Rebuttal\n\n")
            f.write(final_text)
            return gr.update(value=f.name, visible=True)
    
    download_rebuttal_btn.click(
        fn=download_rebuttal,
        inputs=[final_output],
        outputs=[download_rebuttal_file],
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Rebuttal Assistant")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Device for docling PDF processing (cpu or cuda)")
    
    args = parser.parse_args()
    
    device_used = os.environ.get("DOCLING_DEVICE", "cpu")
    
    print(f"\n🚀 Starting AI Rebuttal Assistant")
    print(f"   URL: http://localhost:{args.port}")
    print(f"   Device: {device_used.upper()}")
    print(f"   Share: {'Yes' if args.share else 'No'}\n")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),  # Moved here for Gradio 6.0
        css=APP_CSS,             # Moved here for Gradio 6.0
    )
