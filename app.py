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
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer

# ==========================================
# 0. 页面配置与字体处理 (安全版)
# ==========================================
st.set_page_config(page_title="文风指纹分析实验室 (Pro)", layout="wide")

@st.cache_resource
def get_font_prop():
    font_path = "simhei.ttf"
    if not os.path.exists(font_path):
        return None
    try:
        if os.path.getsize(font_path) / (1024 * 1024) < 1: return None
        return fm.FontProperties(fname=font_path)
    except:
        return None

my_font_prop = get_font_prop()

# ==========================================
# 1. 核心工具函数
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
    if limit: return text[:limit]
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
    # 核心：保留不在黑名单里的词
    text = basic_clean(text)
    words = jieba.lcut(text)
    return [w for w in words if w not in blocklist and not w.isspace()]

def generate_blocklist_from_files(uploaded_files):
    sample_text = ""
    for uploaded_file in uploaded_files:
        content = read_content_safe(uploaded_file)
        sample_text += basic_clean(content)
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

st.title("🕵️‍♂️ 文风分析实验室")
st.markdown("""
上传某位作家的原著，再输入你的同人文本，算法将通过虚词、句式等判断同人文本的还原度，并高亮显示文中哪些词句最具有原著神韵。
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Step 1: 建立基准")
    st.info("请上传原著 TXT 文件（可多选）")
    uploaded_originals = st.file_uploader("上传原著 (支持 .txt)", type="txt", accept_multiple_files=True)

    st.header("Step 2: 输入测试文本")
    fanfic_text = st.text_area("在此粘贴你的同人文本：", height=200, placeholder="建议粘贴 500 字以上的段落...")

    start_btn = st.button("🚀 开始文风分析", type="primary")

# ==========================================
# 3. 主逻辑控制器
# ==========================================

if start_btn:
    if not uploaded_originals or not fanfic_text.strip():
        st.error("请确保已上传原著并输入了测试文本。")
    else:
        with col2:
            status = st.status("正在启动文风解析引擎...", expanded=True)
            
            # --- A: 预处理 ---
            status.write("📖 生成实体停用词表...")
            blocklist = generate_blocklist_from_files(uploaded_originals)
            
            status.write("✂️ 切分与清洗...")
            original_docs = []
            for u_file in uploaded_originals:
                content = read_content_safe(u_file)
                chunks = smart_chunking(content)
                for chunk in chunks:
                    tokens = get_style_tokens(chunk, blocklist)
                    if len(tokens) > 50: original_docs.append(tokens)
            
            # 对同人文本，为了LIME分析，我们最好不要切得太碎，取前 1000 字做演示
            preview_text = fanfic_text[:2000]
            test_tokens = get_style_tokens(preview_text, blocklist)
            
            if len(test_tokens) < 20:
                st.error("测试文本有效词汇不足，无法分析。")
                st.stop()

            # --- B: 训练 FastText ---
            status.write("🧠 训练 FastText 向量空间...")
            # 训练时把测试文本也放进去，建立共享语境
            all_docs = original_docs + [test_tokens]
            model = FastText(sentences=all_docs, vector_size=100, window=5, min_count=1, epochs=20, seed=42)
            
            # --- C: 计算基准向量 ---
            def get_vec(tokens):
                vecs = [model.wv[w] for w in tokens if w in model.wv]
                return np.mean(vecs, axis=0) if vecs else np.zeros(100)

            orig_vecs = np.array([get_vec(d) for d in original_docs])
            gold_standard = np.mean(orig_vecs, axis=0) # 原著质心
            
            # 计算同人分数
            test_vec = get_vec(test_tokens)
            similarity = cosine_similarity([test_vec], [gold_standard])[0][0]
            final_score = similarity * 100
            
            status.update(label="基础分析完成！", state="complete", expanded=False)

            # --- D: 结果展示 ---
            st.divider()
            st.subheader("📊 分析报告")
            
            m1, m2 = st.columns([1, 1])
            with m1:
                st.metric("文风相似度", f"{final_score:.2f}%")
                if final_score > 85: st.success("判定：极度贴合原著")
                elif final_score > 70: st.info("判定：风格较为接近")
                else: st.warning("判定：个人风格强烈")
            
            with m2:
                # 简单的 PCA 可视化
                if len(orig_vecs) > 0:
                    try:
                        pca = PCA(n_components=2)
                        X_all = np.vstack([orig_vecs, [test_vec]])
                        X_pca = pca.fit_transform(X_all)
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.scatter(X_pca[:-1, 0], X_pca[:-1, 1], c='lightgray', s=10, label='Original')
                        center = pca.transform([gold_standard])
                        ax.scatter(center[:,0], center[:,1], c='red', marker='*', s=150, label='Center')
                        ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='blue', marker='X', s=100, label='Fanfic')
                        ax.axis('off')
                        st.pyplot(fig)
                    except: pass

            # ==========================================
            # 4. LIME 可解释性分析 (The "Great Idea")
            # ==========================================
            st.divider()
            st.subheader("🔍 深度归因：为什么像？")
            st.info("LIME 算法将随机遮蔽文本中的词句，观察相似度变化，从而找出对文风贡献最大的片段。")
            
            if st.button("开始 LIME 深度计算 (耗时较长)", type="primary"):
                with st.spinner("正在进行数千次扰动采样，请稍候..."):
                    
                    # 1. 定义 LIME 需要的预测函数
                    # 输入：文本列表 [text1, text2...]
                    # 输出：概率矩阵 [[prob_not_sim, prob_sim], ...]
                    def predict_proba(texts):
                        results = []
                        for text in texts:
                            # 清洗并分词 (使用同样的逻辑)
                            t_tokens = get_style_tokens(text, blocklist)
                            if not t_tokens:
                                results.append([1.0, 0.0]) # 空文本完全不像
                                continue
                            
                            # 获取向量
                            vec = get_vec(t_tokens)
                            # 计算相似度 (0-1)
                            sim = cosine_similarity([vec], [gold_standard])[0][0]
                            # 转换为 [不相似概率, 相似概率]
                            # 为了让 LIME 效果更明显，我们可以对 sim 进行缩放，但原始值也行
                            results.append([1 - sim, sim])
                        return np.array(results)

                    # 2. 初始化解释器
                    # class_names=['Other', 'Original']
                    explainer = LimeTextExplainer(class_names=['差异', '原著风'])

                    # 3. 这里的关键是：LIME 默认按空格分词。
                    # 为了支持中文，我们先把中文文本变成 "词 词 词" 的空格分隔形式
                    # 这样 LIME 就能处理“词”级别的贡献度了
                    seg_list = jieba.cut(preview_text)
                    spaced_text = " ".join(seg_list)

                    # 4. 生成解释
                    # num_features=10: 显示前10个最重要的特征
                    # num_samples=200: 采样次数，越大越准但越慢。云端建议 200-500。
                    exp = explainer.explain_instance(
                        spaced_text, 
                        predict_proba, 
                        num_features=10, 
                        num_samples=200 
                    )

                    # 5. 展示结果 HTML
                    # LIME 会生成一个非常漂亮的 HTML 可视化，包含高亮文本
                    st.write("### 贡献度热力图")
                    components.html(exp.as_html(), height=800, scrolling=True)
                    
                    # 6. 提取具体关键词
                    st.write("### 🏆 最具“原著感”的特征词")
                    st.write("这些词的出现显著提升了文本与原著的相似度（不仅仅是名词，更多是语气词、动词）：")
                    
                    top_features = exp.as_list()
                    # 过滤出正向贡献的词
                    positive_features = [f for f in top_features if f[1] > 0]
                    
                    if positive_features:
                        feat_df = pd.DataFrame(positive_features, columns=["特征词", "贡献度"])
                        st.dataframe(feat_df, use_container_width=True)
                    else:
                        st.write("未检测到显著的正向特征。")

