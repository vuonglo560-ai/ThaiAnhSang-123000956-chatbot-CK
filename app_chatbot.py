# ============================================================
# app_chatbot.py – Chatbot Phân tích Phản hồi Sinh viên (PRO)
# ============================================================

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import re
import os
from typing import List, Dict, Any
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# NLP
try:
    from underthesea import sentiment, word_tokenize
    UNDERTHESEA_OK = True
except ImportError:
    st.error("Vui lòng cài underthesea: pip install underthesea")
    UNDERTHESEA_OK = False
    sentiment = None
    word_tokenize = None

# ============================================================
# CONSTANTS
# ============================================================
EMOJI_MAP = {"positive": "😊", "negative": "😟", "neutral": "😐"}

DEFAULT_STOPWORDS = {
    "và", "của", "là", "các", "cho", "có", "không", "được", "với", "trong",
    "một", "này", "như", "để", "tôi", "bạn", "rất", "nhưng", "cũng", "thì",
    "lại", "nên", "hay", "đang", "sẽ", "đã", "mà", "nếu", "vì", "tại", "khi",
    "ở", "ra", "vào", "lên", "xuống", "đi", "về", "học", "sinh", "viên"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_resource
def load_stopwords() -> set:
    return set(DEFAULT_STOPWORDS)

STOPWORDS = load_stopwords()

def clean_text(text: str) -> str:
    if not text or len(text.strip()) < 2:
        return ""
    text = re.sub(r'[^a-zA-Z0-9\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_emoji_only(text: str) -> bool:
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
    cleaned = re.sub(emoji_pattern, '', text).strip()
    return len(cleaned) == 0

# ============================================================
# INTELLIGENT ANALYSIS (NÂNG CẤP MẠNH)
# ============================================================
def assign_topic(text: str, keywords: List[str]) -> str:
    text_lower = text.lower()
    kw_text = " ".join(keywords).lower()
    
    topic_rules = {
        "Giảng dạy & Phương pháp": ["giảng viên", "thầy", "cô", "bài giảng", "dạy", "phương pháp", "slide"],
        "Chương trình học": ["chương trình", "nội dung", "môn học", "giáo trình", "kiến thức"],
        "Cơ sở vật chất": ["phòng học", "wifi", "máy tính", "điều hòa", "bàn ghế", "cơ sở"],
        "Đánh giá & Thi cử": ["thi", "kiểm tra", "điểm", "chấm", "bài tập", "trắc nghiệm"],
        "Hỗ trợ sinh viên": ["hỗ trợ", "tư vấn", "phòng ban", "thủ tục", "giúp đỡ"],
    }
    
    for topic, keys in topic_rules.items():
        if any(k in text_lower or k in kw_text for k in keys):
            return topic
    return "Khác"

def generate_suggestion(sentiment: str, topic: str) -> str:
    if sentiment != "negative":
        return ""
    suggestions = {
        "Giảng dạy & Phương pháp": "Tăng tương tác, cải thiện slide, dành thời gian trả lời câu hỏi.",
        "Chương trình học": "Cập nhật nội dung thực tiễn, giảm lý thuyết khô khan.",
        "Cơ sở vật chất": "Kiểm tra và nâng cấp wifi, điều hòa, bàn ghế.",
        "Đánh giá & Thi cử": "Làm rõ tiêu chí chấm điểm, đa dạng hình thức đánh giá.",
        "Hỗ trợ sinh viên": "Rút ngắn thời gian xử lý, tăng kênh hỗ trợ trực tuyến.",
    }
    return suggestions.get(topic, "Cần khảo sát thêm để có giải pháp phù hợp.")

def analyze_feedback(text: str) -> Dict[str, Any]:
    if not text or len(text.strip()) < 3:
        return {"sentiment": "neutral", "keywords": [], "confidence": 0.5, "message": "Phản hồi quá ngắn."}

    if is_emoji_only(text):
        return {"sentiment": "positive", "keywords": ["emoji"], "confidence": 0.85, "message": "Emoji-only."}

    cleaned = clean_text(text)
    if not cleaned:
        return {"sentiment": "neutral", "keywords": [], "confidence": 0.4, "message": "Không có nội dung."}

    # Sentiment Analysis - Hybrid
    if UNDERTHESEA_OK:
        try:
            sent_result = sentiment(cleaned)
            raw = str(sent_result).upper()
            sentiment_label = "positive" if "POS" in raw or "TÍCH" in raw else "negative" if "NEG" in raw or "TIÊU" in raw else "neutral"
            base_conf = 0.85
        except:
            sentiment_label = "neutral"
            base_conf = 0.6
    else:
        lower = cleaned.lower()
        pos = sum(w in lower for w in ["tốt","hay","thích","tuyệt","xuất sắc","hài lòng","good","great"])
        neg = sum(w in lower for w in ["kém","tệ","chán","xấu","thất vọng","khó","bad","poor"])
        sentiment_label = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        base_conf = 0.68

    # Keywords
    try:
        tokens = word_tokenize(cleaned, format="text").split() if UNDERTHESEA_OK else re.findall(r'\w+', cleaned)
        keywords = [w.lower() for w in tokens if w.lower() not in STOPWORDS and len(w) > 2]
    except:
        keywords = re.findall(r'\w+', cleaned.lower())

    topic = assign_topic(cleaned, keywords)
    confidence = min(0.95, base_conf + len(keywords)*0.04 + len(cleaned)/600)
    suggestion = generate_suggestion(sentiment_label, topic)

    return {
        "sentiment": sentiment_label,
        "keywords": keywords[:25],
        "confidence": round(confidence, 2),
        "topic": topic,
        "suggestion": suggestion,
        "message": ""
    }

def render_analysis(result: Dict) -> str:
    emoji = EMOJI_MAP.get(result["sentiment"], "😐")
    md = f"""
**{emoji} {result['sentiment'].upper()}** — Độ tin cậy: **{result['confidence']*100:.1f}%**

**Chủ đề:** {result.get('topic', 'Khác')}
**Từ khóa:** {", ".join(result.get('keywords', [])[:12])}
"""
    if result.get("suggestion"):
        md += f"\n\n**💡 Gợi ý cải thiện:** {result['suggestion']}"
    return md.strip()

# ============================================================
# INSIGHTS
# ============================================================
def generate_insights(history: List[Dict]) -> str:
    if len(history) < 3:
        return "Cần thêm dữ liệu để tạo insight."
    
    results = [h["result"] for h in history]
    df = pd.DataFrame(results)
    
    neg_ratio = (df["sentiment"] == "negative").mean() * 100
    hot_topic = df["topic"].value_counts().idxmax() if not df.empty else "Chưa rõ"
    
    return f"""
### 📊 **Insight Thông Minh**
- **Tỷ lệ tiêu cực**: `{neg_ratio:.1f}%` {"⚠️" if neg_ratio > 35 else "✅"}
- **Chủ đề nóng nhất**: **{hot_topic}**
- **Khuyến nghị**: {"Ưu tiên cải thiện " + hot_topic.lower() if neg_ratio > 30 else "Chất lượng tổng thể tốt."}
"""

# ============================================================
# FILE HANDLING, VISUALIZATION, HISTORY... (Giữ nguyên như cũ)
# ============================================================
def handle_file_upload() -> List[str]:
    uploaded = st.sidebar.file_uploader("📁 Upload file phản hồi (CSV/Excel)", type=["csv", "xlsx", "xls"])
    if not uploaded:
        return []
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        
        text_col = next((c for c in ["feedback", "text", "phản_hồi", "phan_hoi", "comment"] if c in df.columns), df.columns[0])
        feedbacks = df[text_col].dropna().astype(str).tolist()
        st.sidebar.success(f"Đã tải {len(feedbacks)} phản hồi.")
        return feedbacks
    except Exception as e:
        st.sidebar.error(f"Lỗi: {e}")
        return []

def export_history(history: List[Dict]) -> bytes:
    if not history:
        return b""
    df = pd.DataFrame([{
        "timestamp": item["timestamp"],
        "feedback": item["feedback"],
        "sentiment": item["result"]["sentiment"],
        "topic": item["result"].get("topic", "Khác"),
        "confidence": item["result"].get("confidence", 0),
        "keywords": ", ".join(item["result"].get("keywords", []))
    } for item in history])
    return df.to_csv(index=False).encode("utf-8")

def render_wordcloud(keywords: List[str]):
    if not keywords:
        return
    try:
        wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(" ".join(keywords))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    except:
        pass

def render_sentiment_timeline(history: List[Dict]):
    if len(history) < 2:
        return
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["score"] = df["result"].apply(lambda x: sentiment_map.get(x.get("sentiment"), 0))
    daily = df.groupby("date")["score"].mean().reset_index()
    fig = px.line(daily, x="date", y="score", title="Xu hướng cảm xúc", markers=True)
    st.plotly_chart(fig, use_container_width=True)

def render_sidebar_stats(history: List[Dict]):
    st.sidebar.header("📊 Thống kê tổng hợp")
    if not history:
        st.sidebar.info("Chưa có dữ liệu.")
        return
    
    sentiments = [h["result"]["sentiment"] for h in history]
    pos, neg, neu = sentiments.count("positive"), sentiments.count("negative"), sentiments.count("neutral")
    
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("😊 Tích cực", pos)
    col2.metric("😟 Tiêu cực", neg)
    col3.metric("😐 Trung lập", neu)

    all_keywords = [kw for h in history for kw in h["result"].get("keywords", [])]
    if all_keywords:
        st.sidebar.subheader("☁️ Word Cloud")
        render_wordcloud(all_keywords)

    if len(history) > 3:
        st.sidebar.subheader("📈 Xu hướng")
        render_sentiment_timeline(history)

# ============================================================
# HISTORY & MAIN
# ============================================================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

def save_history(history: List[Dict], path: str = "history.json"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except:
        pass

def load_history(path: str = "history.json") -> List[Dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def delete_feedback(index: int):
    if 0 <= index < len(st.session_state.history):
        del st.session_state.history[index]
        if len(st.session_state.messages) > index * 2 + 1:
            del st.session_state.messages[index * 2 : index * 2 + 2]
        save_history(st.session_state.history)

def main():
    st.set_page_config(page_title="Phân tích Phản hồi SV Pro", page_icon="🎓", layout="wide")
    init_session_state()

    st.title("🤖 Chatbot Phân tích Phản hồi Sinh viên")
    st.caption("Phân tích thông minh + Gợi ý cải thiện + Insight tự động")

    # Sidebar
    with st.sidebar:
        render_sidebar_stats(st.session_state.history)
        
        uploaded_feedbacks = handle_file_upload()
        for fb in uploaded_feedbacks:
            if fb.strip():
                result = analyze_feedback(fb)
                st.session_state.history.append({"timestamp": datetime.now().isoformat(), "feedback": fb, "result": result})
                st.session_state.messages.append({"role": "user", "content": fb})
                st.session_state.messages.append({"role": "assistant", "content": render_analysis(result)})

        if st.session_state.history and st.sidebar.button("📥 Export CSV"):
            csv_bytes = export_history(st.session_state.history)
            st.sidebar.download_button("Tải file CSV", csv_bytes, f"phan_hoi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    # Chat
    if prompt := st.chat_input("Nhập phản hồi sinh viên..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        for line in [l.strip() for l in prompt.splitlines() if l.strip()]:
            result = analyze_feedback(line)
            analysis_md = render_analysis(result)
            
            st.session_state.history.append({"timestamp": datetime.now().isoformat(), "feedback": line, "result": result})
            st.session_state.messages.append({"role": "assistant", "content": analysis_md})
            
            with st.chat_message("assistant"):
                st.markdown(analysis_md)

        save_history(st.session_state.history)
        st.rerun()

    # Hiển thị lịch sử
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and (i % 2 == 1):
                idx = (i-1)//2
                if st.button("🗑️ Xóa", key=f"del_{idx}"):
                    delete_feedback(idx)
                    st.rerun()

    # Dashboard
    if st.session_state.history:
        with st.expander("📊 Dashboard Phân tích Sâu", expanded=True):
            st.markdown(generate_insights(st.session_state.history))
            
            df_result = pd.DataFrame([h["result"] for h in st.session_state.history])
            tab1, tab2 = st.tabs(["Thống kê", "Word Cloud"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.pie(df_result, names="sentiment", title="Phân bố cảm xúc"), use_container_width=True)
                with c2:
                    st.plotly_chart(px.bar(df_result["topic"].value_counts(), title="Theo chủ đề"), use_container_width=True)

            with tab2:
                all_kw = [kw for res in df_result["keywords"] for kw in res]
                if all_kw:
                    wc = WordCloud(width=900, height=500, background_color="white").generate(" ".join(all_kw))
                    fig, ax = plt.subplots(figsize=(12,6))
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
