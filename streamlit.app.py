# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1XvoIDnmo5CH7adgFcNL6JZTLT-hflLNO")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    labels[0]: {
       "texts": ["흑인", "민머리", "수"],
        "images" : ["https://basketkorea.com/news/data/20140723/p179520446321385_126.jpg"],
        "videos": ["https://www.youtube.com/watch?v=7a6gnRvQqHQ"]
     },

    labels[2]: {
       "texts": ["흑인", "24번", "희색이나 노란색 유나폼 착용"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/watch?v=X0Ju3-10LYI"]
     },

     labels[1]: {
       "texts": ["흑인, 빨간색이나 흰색 유니폼,23번"],
       "images": ["https://image-cdn.hypb.st/https%3A%2F%2Fkr.hypebeast.com%2Ffiles%2F2020%2F04%2Febay-michael-jordan-collectibles-air-jordan-special-launch.jpg?q=75&w=800&cbr=1&fit=max"],
       "videos": ["https://www.youtube.com/watch?v=LLo8BEHmPs4"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
