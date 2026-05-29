# ============================================================
# Chatbot Phân tích Phản hồi Sinh viên – Phiên bản hoàn chỉnh
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from io import BytesIO
import re
import unicodedata
from collections import Counter

# Optional dependencies
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from underthesea import sentiment, word_tokenize, lang_detect
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    sentiment = word_tokenize = lang_detect = None

try:
    from langdetect import detect as langdetect_detect
except ImportError:
    langdetect_detect = None

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================
EMOJI_MAP = {"positive": "😊", "negative": "😟", "neutral": "😐"}

STOPWORDS_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-vi/master/stopwords-vi.txt"

# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dict: {"timestamp", "text", "sentiment", "keywords", "confidence", "lang"}
    if "stopwords" not in st.session_state:
        st.session_state.stopwords = load_stopwords()


def save_history(path: str = "history.json"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_history(path: str = "history.json") -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# ============================================================
# STOPWORDS
# ============================================================
@st.cache_resource
def load_stopwords() -> set:
    """Tải stopwords tiếng Việt từ GitHub (cache resource)."""
    try:
        import requests
        resp = requests.get(STOPWORDS_URL, timeout=10)
        if resp.status_code == 200:
            words = [line.strip() for line in resp.text.splitlines() if line.strip()]
            return set(words)
    except Exception:
        pass
    # Fallback stopwords cơ bản
    return {
        "và", "của", "là", "các", "cho", "có", "không", "được", "với", "trong",
        "một", "này", "để", "rất", "như", "nhưng", "thì", "tôi", "bạn", "học",
        "sinh", "viên", "giảng", "viên", "môn", "học", "lớp", "năm", "thầy", "cô",
        "the", "and", "is", "in", "to", "of", "for", "with", "that", "this", "it", "on"
    }


def extract_keywords(text: str, lang: str) -> list[str]:
    """Trích xuất từ khóa bằng Underthesea hoặc fallback regex."""
    if not text:
        return []

    stopwords = st.session_state.stopwords
    keywords = []

    if UNDERTHESEA_AVAILABLE and word_tokenize and lang == "vi":
        try:
            tokens = word_tokenize(text, format="text").split()
            keywords = [w for w in tokens if w not in stopwords and len(w) > 1]
        except Exception:
            keywords = []

    if not keywords:
        tokens = re.findall(r"[^\W\d_]{2,}", text.lower(), flags=re.UNICODE)
        keywords = [w for w in tokens if w not in stopwords and len(w) > 1]

    return keywords[:20]


def assign_topic(keywords: list[str], text: str) -> str:
    """Gán chủ đề dựa trên từ khóa và nội dung."""
    categories = [
        ("Giảng dạy", ["giảng dạy", "giảng viên", "bài giảng", "giờ học", "học phần"]),
        ("Chương trình", ["chương trình", "nội dung", "môn học", "giáo trình", "khóa học"]),
        ("Cơ sở vật chất", ["phòng", "máy", "thiết bị", "cơ sở", "wifi", "điều hòa", "bàn ghế"]),
        ("Hỗ trợ", ["hỗ trợ", "tư vấn", "phòng ban", "giúp đỡ", "hồ sơ"]),
        ("Dịch vụ", ["dịch vụ", "tổ chức", "thủ tục", "quy trình"]),
    ]
    text_lower = text.lower()
    for label, keys in categories:
        if any(key in text_lower for key in keys) or any(key in keywords for key in keys):
            return label
    return "Khác"


# ============================================================
# LANGUAGE DETECTION
# ============================================================
def detect_language(text: str) -> str:
    """TODO 9: Detect ngôn ngữ."""
    if not text or len(text.strip()) < 3:
        return "vi"

    text = text.strip()
    if UNDERTHESEA_AVAILABLE and lang_detect:
        try:
            return lang_detect(text)
        except:
            pass

    if langdetect_detect:
        try:
            lang = langdetect_detect(text)
            return "vi" if lang in ["vi", "vi"] else lang
        except:
            pass

    # Heuristic đơn giản cho tiếng Việt
    if any(ord(c) > 127 for c in text) and any(c in "ăâđêôơưáàảãạ" for c in text.lower()):
        return "vi"
    return "en"


# ============================================================
# CORE ANALYSIS
# ============================================================
@st.cache_resource
def get_sentiment_model():
    """Cache model underthesea."""
    if UNDERTHESEA_AVAILABLE:
        return sentiment
    return None


def analyze_feedback(text: str) -> dict:
    """TODO 2 + 8 + 13: Phân tích cảm xúc + từ khóa + xử lý edge case."""
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "keywords": [],
            "confidence": 0.0,
            "lang": "unknown",
            "message": "Phản hồi trống"
        }

    text = text.strip()
    lang = detect_language(text)

    # Edge case: quá ngắn hoặc chỉ emoji
    if len(text) <= 3 or re.match(r"^[\U0001F000-\U0001FFFF\s]+$", text):
        sent = "neutral"
        conf = 0.6
        keywords = []
    else:
        model = get_sentiment_model()
        if model and lang == "vi":
            try:
                sent = model(text)
                # underthesea chỉ trả positive/negative/neutral, không có confidence → giả lập
                conf = 0.85 if len(text) > 20 else 0.65
            except:
                sent = "neutral"
                conf = 0.5
        else:
            # Fallback cho tiếng Anh hoặc không có model
            lower = text.lower()
            if any(w in lower for w in ["tốt", "hay", "thích", "tuyệt", "good", "great", "excellent"]):
                sent = "positive"
            elif any(w in lower for w in ["kém", "tệ", "chán", "xấu", "bad", "poor", "terrible"]):
                sent = "negative"
            else:
                sent = "neutral"
            conf = 0.7

    keywords = extract_keywords(text, lang)
    topic = assign_topic(keywords, text)

    return {
        "sentiment": sent,
        "keywords": keywords,  # giới hạn đã xử lý trong extract_keywords
        "confidence": round(conf, 2),
        "lang": lang,
        "topic": topic,
        "message": ""
    }


def render_analysis(result: dict) -> str:
    """Tạo markdown đẹp cho chat bubble."""
    emoji = EMOJI_MAP.get(result["sentiment"], "😐")
    conf_pct = int(result["confidence"] * 100)

    md = f"""
**Cảm xúc:** {emoji} **{result['sentiment'].upper()}** (Confidence: {conf_pct}%)  
**Ngôn ngữ:** {result['lang'].upper()}
"""
    if result["keywords"]:
        md += f"**Từ khóa chính:** {', '.join(result['keywords'][:8])}\n"

    if result.get("message"):
        md += f"\n> {result['message']}"

    return md.strip()


# ============================================================
# FILE HANDLING
# ============================================================
def handle_file_upload() -> list[str]:
    """TODO 3: Hỗ trợ upload CSV/Excel."""
    uploaded = st.sidebar.file_uploader("📤 Upload file phản hồi (CSV/Excel)", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            # Tìm cột chứa phản hồi (ưu tiên: feedback, comment, nội dung, text, column 0)
            col = None
            for c in ["feedback", "comment", "nội dung", "text", "phản hồi"]:
                if c.lower() in [x.lower() for x in df.columns]:
                    col = c
                    break
            if col is None and not df.empty:
                col = df.columns[0]

            if col:
                return df[col].dropna().astype(str).tolist()
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file: {e}")
    return []


def export_history(history: list[dict]) -> bytes:
    """Export sang CSV."""
    if not history:
        return b""
    df = pd.DataFrame(history)
    return df.to_csv(index=False).encode("utf-8-sig")


def export_history_excel(history: list[dict]) -> bytes:
    """Export lịch sử sang XLSX."""
    if not history:
        return b""
    df = pd.DataFrame(history)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="History")
    return output.getvalue()


def export_history_pdf(history: list[dict]) -> bytes:
    """Export lịch sử sang PDF."""
    if not history or not PDF_AVAILABLE:
        return b""

    def _safe_pdf_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return stripped.encode("latin-1", errors="replace").decode("latin-1")

    def _find_bold_font_path(candidate_path: str) -> str:
        folder = os.path.dirname(candidate_path)
        basename = os.path.splitext(os.path.basename(candidate_path))[0]
        candidates = [
            os.path.join(folder, f"{basename}-Bold.ttf"),
            os.path.join(folder, f"{basename}Bold.ttf"),
            os.path.join(folder, "DejaVuSans-Bold.ttf"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidate_path

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Load a Unicode font file for Vietnamese text if available.
    repo_font = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
    font_path_candidates = [
        repo_font,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    font_path = None
    for candidate in font_path_candidates:
        if candidate and os.path.exists(candidate):
            font_path = candidate
            break

    bold_available = False
    if font_path:
        try:
            pdf.add_font("DejaVu", "", font_path, uni=True)
            bold_font_path = _find_bold_font_path(font_path)
            try:
                pdf.add_font("DejaVu", "B", bold_font_path, uni=True)
                bold_available = True
            except Exception:
                bold_available = False
        except Exception:
            font_path = None
            bold_available = False

    if font_path:
        title_font = "DejaVu"
        body_font = "DejaVu"
    else:
        title_font = "Arial"
        body_font = "Arial"

    title_style = "B" if bold_available else ""
    header_style = "B" if bold_available else ""

    pdf.set_font(title_font, title_style, 14)
    pdf.cell(0, 10, "Báo cáo phân tích phản hồi", ln=True)
    pdf.set_font(body_font, size=10)
    pdf.ln(2)

    for index, item in enumerate(history, 1):
        text = item.get("text", "")
        topic = item.get("topic", "Khác")
        sentiment = item.get("sentiment", "neutral").upper()
        lang = item.get("lang", "unknown")
        confidence = int(item.get("confidence", 0) * 100)

        pdf.set_font(body_font, header_style, 10)
        pdf.multi_cell(0, 6, f"{index}. [{sentiment}] {topic} ({lang}, {confidence}%)")
        pdf.set_font(body_font, size=10)
        pdf.multi_cell(0, 6, text)
        pdf.ln(2)

    try:
        return pdf.output(dest="S").encode("latin-1")
    except Exception:
        fallback_pdf = FPDF()
        fallback_pdf.set_auto_page_break(auto=True, margin=15)
        fallback_pdf.add_page()
        fallback_pdf.set_font("Arial", size=10)
        fallback_pdf.cell(0, 10, _safe_pdf_text("Báo cáo phân tích phản hồi"), ln=True)
        fallback_pdf.set_font("Arial", size=10)
        fallback_pdf.ln(2)

        for index, item in enumerate(history, 1):
            text = _safe_pdf_text(item.get("text", ""))
            topic = _safe_pdf_text(item.get("topic", "Khác"))
            sentiment = item.get("sentiment", "neutral").upper()
            lang = _safe_pdf_text(item.get("lang", "unknown"))
            confidence = int(item.get("confidence", 0) * 100)

            fallback_pdf.multi_cell(0, 6, _safe_pdf_text(f"{index}. [{sentiment}] {topic} ({lang}, {confidence}%)"))
            fallback_pdf.multi_cell(0, 6, text)
            fallback_pdf.ln(2)

        return fallback_pdf.output(dest="S").encode("latin-1")


# ============================================================
# VISUALIZATION
# ============================================================
def render_wordcloud(keywords_list: list):
    """TODO 5: Word cloud từ tất cả keywords trong history."""
    if not WORDCLOUD_AVAILABLE or not keywords_list:
        st.info("Cần cài `wordcloud` và `matplotlib` để hiển thị word cloud.")
        return

    all_words = []
    for kws in keywords_list:
        all_words.extend(kws)

    if not all_words:
        return

    text = " ".join(all_words)
    try:
        wc = WordCloud(width=800, height=400, background_color="white", max_words=100, colormap="viridis").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except Exception:
        st.warning("Không thể tạo word cloud.")


def render_sentiment_timeline(history: list[dict]):
    """Biểu đồ xu hướng cảm xúc theo thời gian (score đơn giản)."""
    if len(history) < 2:
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["score"] = df["sentiment"].map(sentiment_map)

    st.line_chart(df.set_index("timestamp")["score"], use_container_width=True)


def render_sentiment_trend_history(history: list[dict], period: str = "D"):
    """Biểu đồ nâng cao: số phản hồi theo cảm xúc theo chu kỳ thời gian."""
    if len(history) < 2:
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["period"] = df["timestamp"].dt.to_period(period).dt.to_timestamp()
    df["count"] = 1

    trend = df.pivot_table(index="period", columns="sentiment", values="count", aggfunc="sum", fill_value=0)
    if trend.empty:
        return
    st.line_chart(trend, use_container_width=True)


def render_top_keywords(history: list[dict], limit: int = 10) -> list[tuple[str, int]]:
    all_keywords = [kw for h in history for kw in h.get("keywords", [])]
    if not all_keywords:
        return []
    return Counter(all_keywords).most_common(limit)


def render_sidebar_stats(history: list[dict], period: str = "D"):
    """Sidebar thống kê + TODO 5 + 12."""
    st.sidebar.header("📊 Thống kê tổng hợp")

    if not history:
        st.sidebar.info("Chưa có dữ liệu phân tích.")
        return

    sentiments = [h["sentiment"] for h in history]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")

    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("😊 Positive", pos)
    col2.metric("😟 Negative", neg)
    col3.metric("😐 Neutral", neu)

    st.sidebar.subheader("📈 Phân bố cảm xúc")
    df_counts = pd.DataFrame({"sentiment": ["positive", "neutral", "negative"], "count": [pos, neu, neg]})
    st.sidebar.bar_chart(df_counts.set_index("sentiment"))

    st.sidebar.subheader("☁️ Top từ khóa")
    top_keywords = render_top_keywords(history, 8)
    if top_keywords:
        for kw, count in top_keywords:
            st.sidebar.write(f"- **{kw}** ({count})")
    else:
        st.sidebar.write("Không có từ khóa rõ ràng.")

    st.sidebar.subheader("☁️ Word Cloud")
    all_keywords = [h.get("keywords", []) for h in history]
    render_wordcloud(all_keywords)

    st.sidebar.subheader("📈 Xu hướng cảm xúc nâng cao")
    render_sentiment_trend_history(history, period)


# ============================================================
# HISTORY MANAGEMENT
# ============================================================
def delete_feedback(index: int):
    """TODO 7: Xóa phản hồi."""
    if 0 <= index < len(st.session_state.history):
        del st.session_state.history[index]
        # Cũng xóa message tương ứng (phức tạp một chút)
        if len(st.session_state.messages) > index * 2 + 1:  # user + assistant
            del st.session_state.messages[index * 2 : index * 2 + 2]
        save_history()


def filter_history(history: list[dict], sentiment_filter: list[str], query: str) -> list[dict]:
    if not history:
        return []
    if not sentiment_filter:
        sentiment_filter = ["positive", "neutral", "negative"]
    filtered = [h for h in history if h["sentiment"] in sentiment_filter]
    if query:
        q = query.lower()
        filtered = [h for h in filtered if q in h.get("text", "").lower() or q in " ".join(h.get("keywords", [])).lower()]
    return filtered


def render_filtered_summary(filtered: list[dict]):
    if not filtered:
        st.info("Không có phản hồi phù hợp với bộ lọc hiện tại.")
        return

    st.subheader("Tổng quan bộ lọc")
    st.write(f"Có {len(filtered)} phản hồi phù hợp với điều kiện hiện tại.")

    sentiments = [item["sentiment"] for item in filtered]
    counts = {"positive": sentiments.count("positive"), "neutral": sentiments.count("neutral"), "negative": sentiments.count("negative")}

    c1, c2, c3 = st.columns(3)
    c1.metric("😊 Positive", counts["positive"])
    c2.metric("😐 Neutral", counts["neutral"])
    c3.metric("😟 Negative", counts["negative"])

    if counts["positive"] + counts["neutral"] + counts["negative"] > 0:
        st.bar_chart(pd.DataFrame({"count": list(counts.values())}, index=list(counts.keys())))

    topic_counts = Counter([item.get("topic", "Khác") for item in filtered])
    if topic_counts:
        st.subheader("📌 Nhóm chủ đề")
        for topic, count in topic_counts.most_common(6):
            st.write(f"- **{topic}**: {count}")


# ============================================================
# HELP PAGE
# ============================================================
def render_help_page():
    """TODO 10: Hướng dẫn sử dụng."""
    with st.expander("📖 Hướng dẫn sử dụng & Giải thích chỉ số"):
        st.markdown("""
### Cách sử dụng
- Nhập phản hồi trực tiếp vào ô chat (có thể nhiều dòng).
- Hoặc upload file CSV/Excel ở sidebar.
- Xem kết quả phân tích cảm xúc + từ khóa ngay trong chat.
- Sidebar hiển thị thống kê tổng hợp, word cloud và timeline.

### Ý nghĩa chỉ số
- **Cảm xúc**: positive / negative / neutral (dựa trên underthesea cho tiếng Việt).
- **Confidence**: Độ tin cậy của kết quả (0.0 - 1.0).
- **Từ khóa**: Các từ quan trọng sau khi loại stopwords.

**Lưu ý**: Ứng dụng hỗ trợ tốt nhất với tiếng Việt.
        """)


# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(page_title="Chatbot Phân tích Phản hồi SV", page_icon="🤖", layout="wide")

    init_session_state()

    # Load history từ file nếu có
    if not st.session_state.history:
        st.session_state.history = load_history()

    # ── Sidebar ──
    with st.sidebar:
        period = st.selectbox(
            "Chu kỳ biểu đồ thời gian",
            ["D", "W", "M"],
            format_func=lambda x: {"D": "Hàng ngày", "W": "Hàng tuần", "M": "Hàng tháng"}[x],
            index=0
        )
        render_sidebar_stats(st.session_state.history, period)

        st.sidebar.markdown("---")
        st.sidebar.header("🔎 Bộ lọc dữ liệu")
        sentiment_filter = st.sidebar.multiselect(
            "Lọc cảm xúc",
            ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"]
        )
        search_query = st.sidebar.text_input("Tìm kiếm phản hồi / từ khóa")

        if st.sidebar.button("🧹 Xóa toàn bộ lịch sử"):
            st.session_state.history = []
            st.session_state.messages = []
            save_history()
            st.sidebar.success("Đã xóa toàn bộ lịch sử.")
            st.rerun()

        st.divider()
        uploaded_texts = handle_file_upload()
        if uploaded_texts:
            st.sidebar.success(f"Đã tải {len(uploaded_texts)} phản hồi từ file.")
            if st.sidebar.button("Phân tích tất cả từ file"):
                for txt in uploaded_texts:
                    result = analyze_feedback(txt)
                    result["timestamp"] = datetime.now().isoformat()
                    result["text"] = txt

                    st.session_state.history.append(result)

                    # Thêm vào messages
                    st.session_state.messages.append({"role": "user", "content": txt})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": render_analysis(result)
                    })

                save_history()
                st.rerun()

        # Export
        if st.session_state.history:
            csv_bytes = export_history(st.session_state.history)
            st.sidebar.download_button(
                label="Tải file CSV",
                data=csv_bytes,
                file_name=f"phan_tich_phan_hoi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            excel_bytes = export_history_excel(st.session_state.history)
            st.sidebar.download_button(
                label="Tải file XLSX",
                data=excel_bytes,
                file_name=f"phan_tich_phan_hoi_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if PDF_AVAILABLE:
                pdf_bytes = export_history_pdf(st.session_state.history)
                st.sidebar.download_button(
                    label="Tải file PDF",
                    data=pdf_bytes,
                    file_name=f"phan_tich_phan_hoi_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.sidebar.info("Cần cài `fpdf` để xuất PDF.")

        render_help_page()

    # ── Main Area ──
    st.title("🤖 Chatbot Phân tích Phản hồi Sinh viên")

    filtered_history = filter_history(st.session_state.history, sentiment_filter, search_query)

    with st.expander("📌 Tổng quan phân tích", expanded=True):
        render_filtered_summary(filtered_history)
        if filtered_history:
            df_view = pd.DataFrame(filtered_history)
            df_view["topic"] = df_view.get("topic", "Khác")
            df_view = df_view[["timestamp", "text", "topic", "sentiment", "confidence", "lang"]]
            df_view["timestamp"] = pd.to_datetime(df_view["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(df_view.sort_values("timestamp", ascending=False).reset_index(drop=True), use_container_width=True)

    # Hiển thị lịch sử chat
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Nút xóa (chỉ cho user message)
            if msg["role"] == "user" and i // 2 < len(st.session_state.history):
                if st.button("🗑️ Xóa", key=f"del_{i}"):
                    delete_feedback(i // 2)
                    st.rerun()

    # Ô nhập chat
    if prompt := st.chat_input("Nhập phản hồi của sinh viên tại đây... (có thể nhiều dòng)"):
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]

        for line in lines:
            # Lưu user message
            st.session_state.messages.append({"role": "user", "content": line})

            with st.chat_message("user"):
                st.markdown(line)

            # Phân tích
            result = analyze_feedback(line)
            result["timestamp"] = datetime.now().isoformat()
            result["text"] = line

            st.session_state.history.append(result)

            # Hiển thị kết quả
            analysis_md = render_analysis(result)
            st.session_state.messages.append({"role": "assistant", "content": analysis_md})

            with st.chat_message("assistant"):
                st.markdown(analysis_md)

        save_history()
        st.rerun()


if __name__ == "__main__":
    main()