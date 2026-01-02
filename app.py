#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import jieba
import jieba.posseg as pseg
from gensim.models import FastText
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import re
import os

# ==========================================
# 0. 页面配置与字体处理 (超级安全版)
# ==========================================
st.set_page_config(page_title="文风指纹分析实验室", layout="wide")

@st.cache_resource
def get_font_prop():
    """
    只返回 FontProperties 对象，不尝试读取文件内部信息，
    避免因文件损坏导致 get_name() 崩溃。
    """
    font_path = "simhei.ttf"
    
    # 1. 检查文件是否存在
    if not os.path.exists(font_path):
        st.warning(f"⚠️ 警告：未找到 '{font_path}'。图表中文可能无法显示。")
        return None
    
    # 2. 检查文件大小 (防止空文件)
    try:
        file_size_mb = os.path.getsize(font_path) / (1024 * 1024)
        if file_size_mb < 1:
            st.warning(f"⚠️ 字体文件异常：'{font_path}' 只有 {file_size_mb:.2f} MB (正常应>9MB)。这可能是一个损坏的文件或HTML页面。图表中文可能显示为方块。")
            return None
    except Exception:
        return None

    # 直接返回 property，不进行任何读取操作
    return fm.FontProperties(fname=font_path)

# 获取字体属性对象
my_font_prop = get_font_prop()

# 【关键修改】不再设置 plt.rcParams['font.family']
# 因为这步操作会强制读取字体文件头，如果文件坏了就会直接崩溃。
# 我们改为在绘图时单独指定 fontproperties。

# ==========================================
# 1. 核心工具函数：智能编码读取
# ==========================================

def read_content_safe(file_obj, limit=None):
    try:
        file_obj.seek(0)
        content_bytes = file_obj.read()
        text = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = content_bytes.decode('gbk')
        except UnicodeDecodeError:
            text = content_bytes.decode('utf-8', errors='ignore')
    if limit:
        return text[:limit]
    return text

def basic_clean(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'第.+?章.*', '', text)
    text = re.sub(r'Chapter.*', '', text)
    text = re.sub(r'\[\d+\]|[\u2460-\u24FF]', '', text)
    punctuation_map = {',': '，', '!': '！', '?': '？', '(': '（', ')': '）', ':': '：', ';': '；'}
    for eng_punc, chi_punc in punctuation_map.items():
        text = text.replace(eng_punc, chi_punc)
    return text

def smart_chunking(text, min_length=300):
    lines = text.split("\n")
    final_chunks = []
    current_chunk = ""
    for line in lines:
        line = line.strip()
        if len(line) < 2: continue
        current_chunk += line + " "
        if len(current_chunk) >= min_length:
            final_chunks.append(current_chunk)
            current_chunk = ""
    if len(current_chunk) > 50:
        final_chunks.append(current_chunk)
    return final_chunks

def get_style_tokens(text, blocklist):
    text = basic_clean(text)
    words = jieba.lcut(text)
    return [w for w in words if w not in blocklist and not w.isspace()]

def generate_blocklist_from_files(uploaded_files):
    sample_text = ""
    for uploaded_file in uploaded_files:
        content = read_content_safe(uploaded_file)
        sample_text += basic_clean(content)[:50000]
    
    words = pseg.cut(sample_text)
    candidates = []
    target_flags = {'nr', 'ns', 'nz', 'nt', 'per', 'loc'}
    for w, f in words:
        if len(w) > 1 and f in target_flags:
            candidates.append(w)
    from collections import Counter
    blocklist = set([w for w, c in Counter(candidates).most_common(500)])
    return blocklist

# ==========================================
# 2. 网站界面 UI
# ==========================================

st.title("🕵️‍♂️ 文风指纹分析实验室")
st.markdown("""
这是一个基于 **FastText** 和 **Stylometry (文体学)** 的分析工具。
上传某位作家的原著，再输入你的同人文本，算法将自动剥离“内容”，仅根据“文风”计算相似度。
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Step 1: 建立基准")
    st.info("请上传原著 TXT 文件（可多选）。支持 UTF-8 和 GBK 编码。")
    uploaded_originals = st.file_uploader("上传原著 (支持 .txt)", type="txt", accept_multiple_files=True)

    st.header("Step 2: 输入测试文本")
    fanfic_text = st.text_area("在此粘贴你的同人/测试文本：", height=200, placeholder="把要测试的小说片段粘贴在这里...")

    start_btn = st.button("🚀 开始文风分析", type="primary")

# ==========================================
# 3. 主逻辑控制器
# ==========================================

if start_btn:
    if not uploaded_originals:
        st.error("请先上传原著文件！")
    elif not fanfic_text.strip():
        st.error("请输入测试文本！")
    else:
        with col2:
            with st.status("正在进行深度分析...", expanded=True) as status:
                
                # --- A: 预处理 ---
                st.write("📖 读取原著并生成黑名单...")
                blocklist = generate_blocklist_from_files(uploaded_originals)
                st.write(f"✅ 已屏蔽 {len(blocklist)} 个高频专名")

                st.write("✂️ 智能分段中...")
                original_docs = []
                for u_file in uploaded_originals:
                    content = read_content_safe(u_file)
                    chunks = smart_chunking(content)
                    for chunk in chunks:
                        tokens = get_style_tokens(chunk, blocklist)
                        if len(tokens) > 50:
                            original_docs.append(tokens)
                
                # --- B: 测试文本 ---
                test_docs = []
                test_chunks = smart_chunking(fanfic_text, min_length=200)
                for chunk in test_chunks:
                    tokens = get_style_tokens(chunk, blocklist)
                    if len(tokens) > 50:
                        test_docs.append(tokens)
                
                if not test_docs:
                    st.error("测试文本有效词汇太少。")
                    st.stop()

                # --- C: 训练 ---
                st.write("🧠 训练 FastText 模型...")
                all_docs = original_docs + test_docs
                model = FastText(sentences=all_docs, vector_size=100, window=5, min_count=1, epochs=20, seed=42)
                
                # --- D: 计算 ---
                def get_vec(tokens):
                    vecs = [model.wv[w] for w in tokens if w in model.wv]
                    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

                orig_vecs = np.array([get_vec(d) for d in original_docs])
                test_vecs = np.array([get_vec(d) for d in test_docs])
                
                gold_standard = np.mean(orig_vecs, axis=0)
                test_centroid = np.mean(test_vecs, axis=0)
                
                similarity = cosine_similarity([test_centroid], [gold_standard])[0][0]
                final_score = similarity * 100
                
                status.update(label="分析完成！", state="complete", expanded=False)

            # ==========================================
            # 4. 结果展示
            # ==========================================
            st.divider()
            st.subheader("分析结果")
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label="文风相似度", value=f"{final_score:.2f}%")
                if final_score > 90:
                    st.success("判定：极度相似")
                elif final_score > 75:
                    st.info("判定：风格接近")
                else:
                    st.warning("判定：差异显著")

            with metric_col2:
                st.write("### 向量空间投影")
                if len(orig_vecs) > 0:
                    try:
                        pca = PCA(n_components=2)
                        X_all = np.vstack([orig_vecs, test_vecs])
                        X_pca = pca.fit_transform(X_all)
                        n_orig = len(orig_vecs)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # 绘图部分
                        ax.scatter(X_pca[:n_orig, 0], X_pca[:n_orig, 1], c='lightgray', s=10, alpha=0.5, label='原著切片')
                        center = pca.transform([gold_standard])
                        ax.scatter(center[:,0], center[:,1], c='red', marker='*', s=200, label='原著基准')
                        ax.scatter(X_pca[n_orig:, 0], X_pca[n_orig:, 1], c='blue', s=80, marker='X', label='你的文本')
                        
                        # 【安全绘图】只有当字体对象有效时，才应用字体
                        if my_font_prop:
                            ax.legend(prop=my_font_prop)
                            ax.set_title("文风落点分布", fontproperties=my_font_prop)
                        else:
                            # 字体坏了就用默认字体（英文），防止崩溃
                            ax.legend()
                            ax.set_title("Style Distribution (Font Missing)")
                            
                        ax.axis('off')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"绘图出错: {e}")

