import streamlit as st
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Victorian Story Generator",
    page_icon="📖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Title */
.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    color: #666;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Output box (FIXED TEXT COLOR ISSUE) */
.output-box {
    background-color: #f7f7f7;
    color: #1a1a1a;   /* IMPORTANT FIX */
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #ddd;
    white-space: pre-wrap;
    line-height: 1.7;
    font-size: 16px;
}

/* Buttons spacing */
div.stButton > button {
    height: 45px;
    font-size: 16px;
    border-radius: 10px;
}

/* Sidebar look */
[data-testid="stSidebar"] {
    background-color: #fafafa;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "generated_story" not in st.session_state:
    st.session_state.generated_story = ""

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">📖 Victorian Story Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Generate elegant Victorian-style stories using your custom-trained language model</div>',
    unsafe_allow_html=True
)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2.2, 1])

with col1:
    prompt = st.text_area(
        "Story Prompt",
        placeholder="Example: A lonely gentleman discovers a mysterious letter in London...",
        height=220
    )

with col2:
    st.subheader("⚙ Settings")

    temperature = st.slider(
        "Creativity (Temperature)",
        0.1, 2.0, 1.0, 0.1,
        help="Higher = more creative, Lower = more focused"
    )

    max_tokens = st.slider(
        "Max Length (Tokens)",
        50, 1000, 300, 50
    )

# ---------------- BUTTONS ----------------
btn1, btn2 = st.columns(2)

generate_clicked = btn1.button("✨ Generate Story", use_container_width=True)
clear_clicked = btn2.button("🗑 Clear", use_container_width=True)

# Clear
if clear_clicked:
    st.session_state.generated_story = ""
    st.rerun()

# ---------------- GENERATION ----------------
if generate_clicked:

    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("📜 Writing your Victorian masterpiece..."):

            try:
                response = requests.post(
                    "http://127.0.0.1:5000/generate",
                    json={
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.generated_story = result["generated_text"]
                    st.success("Story generated successfully!")

                else:
                    st.error(f"Server Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Backend not running. Start your Flask server first.")

            except requests.exceptions.Timeout:
                st.error("Request timed out. Model is taking too long.")

            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# ---------------- OUTPUT ----------------
if st.session_state.generated_story:

    st.subheader("📖 Generated Story")

    st.markdown(
        f"""
        <div class="output-box">
        {st.session_state.generated_story}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇ Download Story",
        st.session_state.generated_story,
        file_name="victorian_story.txt",
        mime="text/plain"
    )