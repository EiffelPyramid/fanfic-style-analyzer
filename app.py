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

# 引入 LIME 库
from lime.lime_text import LimeTextExplainer

# ==========================================
# 0. 页面配置与字体安全检查
# ==========================================
st.set_page_config(page_title="文风指纹分析实验室 (终极版)", layout="wide")

@st.cache_resource
def get_font_prop():
    font_path = "simhei.ttf"
    # 严格检查字体文件
    if not os.path.exists(font_path): return None
    try:
        if os.path.getsize(font_path) / (1024 * 1024) < 1: return None
        return fm.FontProperties(fname=font_path)
    except: return None

my_font_prop = get_font_prop()

# ==========================================
# 1. 核心工具函数
# ==========================================

def read_content_safe(file_obj, limit=None):
    """安全读取文件内容 (兼容 UTF-8 和 GBK)"""
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
    """基础清洗：去章节头、统一标点"""
    if not isinstance(text, str): return ""
    text = re.sub(r'第.+?章.*', '', text)
    text = re.sub(r'Chapter.*', '', text)
    text = re.sub(r'\[\d+\]|[\u2460-\u24FF]', '', text)
    punctuation_map = {',': '，', '!': '！', '?': '？', '(': '（', ')': '）', ':': '：', ';': '；'}
    for eng_punc, chi_punc in punctuation_map.items():
        text = text.replace(eng_punc, chi_punc)
    return text

def smart_chunking(text, min_length=300):
    """智能分段"""
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
    """文风分词：基于黑名单过滤内容词"""
    text = basic_clean(text)
    words = jieba.lcut(text)
    return [w for w in words if w not in blocklist and not w.isspace()]

def generate_blocklist_from_files(uploaded_files):
    """自动生成内容词黑名单"""
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

st.title("🕵️‍♂️ 文风指纹分析实验室 (Pro Plus)")
st.markdown("""
本系统通过 **FastText** 向量化与 **LIME** 可解释性模型，对文本进行双重分析：
1.  **文风指纹比对**：剥离剧情内容，仅通过虚词、句式等“指纹”计算整体相似度。
2.  **深度归因解释**：高亮显示文本中哪些词句对“像原著”贡献最大。
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Step 1: 建立基准")
    uploaded_originals = st.file_uploader("上传原著 (支持 .txt)", type="txt", accept_multiple_files=True)
    
    st.header("Step 2: 输入测试文本")
    fanfic_text = st.text_area("在此粘贴同人/测试文本：", height=250, placeholder="建议粘贴 500 字以上的段落...")

    # 这里的按钮一旦点击，就会触发下面的所有逻辑
    start_btn = st.button("🚀 一键开启全流程分析", type="primary")

# ==========================================
# 3. 主逻辑控制器 (合并了基础与进阶)
# ==========================================

if start_btn:
    if not uploaded_originals:
        st.error("❌ 请先上传原著文件！")
    elif not fanfic_text.strip():
        st.error("❌ 请输入测试文本！")
    else:
        with col2:
            # === 阶段一：基础模型构建与计算 ===
            with st.status("正在进行全流程分析...", expanded=True) as status:
                
                # 1. 预处理
                status.write("📖 正在扫描原著并构建去噪黑名单...")
                blocklist = generate_blocklist_from_files(uploaded_originals)
                status.write(f"✅ 已屏蔽 {len(blocklist)} 个高频专有名词（如：{list(blocklist)[:3]}...）")
                
                # 2. 数据切分
                status.write("✂️ 正在进行文本切片与清洗...")
                original_docs = []
                for u_file in uploaded_originals:
                    content = read_content_safe(u_file)
                    chunks = smart_chunking(content)
                    for chunk in chunks:
                        tokens = get_style_tokens(chunk, blocklist)
                        if len(tokens) > 50: original_docs.append(tokens)
                
                # 处理同人文本
                preview_text = fanfic_text[:2000] # 取前2000字做深度分析
                test_tokens = get_style_tokens(preview_text, blocklist)
                
                if len(test_tokens) < 20:
                    status.update(label="分析失败：有效词汇不足", state="error")
                    st.stop()

                # 3. 训练模型
                status.write("🧠 正在训练 FastText 文风向量空间...")
                all_docs = original_docs + [test_tokens]
                model = FastText(sentences=all_docs, vector_size=100, window=5, min_count=1, epochs=20, seed=42)
                
                # 4. 计算相似度
                def get_vec(tokens):
                    vecs = [model.wv[w] for w in tokens if w in model.wv]
                    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

                orig_vecs = np.array([get_vec(d) for d in original_docs])
                gold_standard = np.mean(orig_vecs, axis=0) # 原著质心
                test_vec = get_vec(test_tokens)
                
                similarity = cosine_similarity([test_vec], [gold_standard])[0][0]
                final_score = similarity * 100
                
                status.write("✅ 基础分析完成！")
                status.update(label="第一阶段分析完成，正在进行 LIME 深度归因...", state="running", expanded=True)

                # === 阶段二：结果展示 (基础部分) ===
                # 这里就是你希望“保留前一条输出”的地方，我把它放回来了
                
                st.divider()
                st.subheader("📊 基础分析报告")
                
                res_c1, res_c2 = st.columns([1, 1])
                
                with res_c1:
                    st.metric(label="整体文风相似度", value=f"{final_score:.2f}%")
                    
                    # 详细评语逻辑 (恢复你喜欢的文字说明)
                    if final_score > 90:
                        st.success("""
                        **判定：极度相似（Tier S）**
                        这段文本在虚词使用、句式节奏和用词习惯上与原著高度一致。
                        机器认为这极有可能是原作者本人或极其资深的模仿者所写。
                        """)
                    elif final_score > 75:
                        st.info("""
                        **判定：风格接近（Tier A）**
                        文本抓住了原著的语感特征，但在部分细节处理上仍有个人色彩。
                        这是一个非常优秀的同人创作，读起来很有“那味儿”。
                        """)
                    elif final_score > 60:
                        st.warning("""
                        **判定：略有差异（Tier B）**
                        虽然属于同人范畴，但作者保留了强烈的个人叙述风格。
                        文风与原著有明显区别（可能是OOC或AU设定导致）。
                        """)
                    else:
                        st.error("""
                        **判定：差异显著（Tier C）**
                        机器难以识别出这是基于原著的仿写。这可能是一篇完全架空的现代文，
                        或者作者的写作习惯与原著大相径庭。
                        """)

                with res_c2:
                    st.write("**向量空间投影 (PCA)**")
                    if len(orig_vecs) > 0:
                        try:
                            pca = PCA(n_components=2)
                            X_all = np.vstack([orig_vecs, [test_vec]])
                            X_pca = pca.fit_transform(X_all)
                            
                            fig, ax = plt.subplots(figsize=(6, 4))
                            # 原著点（背景）
                            ax.scatter(X_pca[:-1, 0], X_pca[:-1, 1], c='lightgray', s=15, alpha=0.6, label='原著切片')
                            # 原著中心
                            center = pca.transform([gold_standard])
                            ax.scatter(center[:,0], center[:,1], c='red', marker='*', s=200, label='原著基准')
                            # 测试文本点
                            ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='blue', s=100, marker='X', edgecolors='white', label='你的文本')
                            
                            # 字体安全设置
                            if my_font_prop:
                                ax.legend(prop=my_font_prop)
                                ax.set_title("文风落点分布图", fontproperties=my_font_prop)
                            else:
                                ax.legend()
                                ax.set_title("Style Distribution")
                                
                            ax.axis('off') # 去掉坐标轴更美观
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"绘图错误: {e}")

                # === 阶段三：LIME 进阶分析 (自动继续执行) ===
                st.divider()
                st.subheader("🔍 进阶分析：LIME 可解释性归因")
                st.info("AI 正在通过随机遮蔽实验，寻找文中对“原著感”贡献最大的句子... (这可能需要十几秒)")
                
                # 进度条
                lime_progress = st.progress(0)
                
                # 1. 定义 LIME 预测函数 (桥接 FastText)
                def predict_proba(texts):
                    results = []
                    # 模拟进度：这只是个简单的 trick，因为 predict_proba 会被调用几百次
                    # 实际很难精确控制进度条，这里只能显示“正在计算”
                    for text in texts:
                        t_tokens = get_style_tokens(text, blocklist)
                        if not t_tokens:
                            results.append([1.0, 0.0])
                            continue
                        vec = get_vec(t_tokens)
                        sim = cosine_similarity([vec], [gold_standard])[0][0]
                        # 放大差异以便 LIME 更好捕捉：(sim^3 增加对比度)
                        sim_scaled = sim ** 3 
                        results.append([1 - sim_scaled, sim_scaled])
                    return np.array(results)

                # 2. 初始化解释器
                explainer = LimeTextExplainer(class_names=['差异', '原著风'])

                # 3. 中文分词适配 (关键步骤)
                # LIME 需要空格分隔的字符串
                seg_list = jieba.cut(preview_text)
                spaced_text = " ".join(seg_list)

                # 4. 生成解释 (减少采样数以加快速度)
                # num_samples=200 足够演示用
                try:
                    exp = explainer.explain_instance(
                        spaced_text, 
                        predict_proba, 
                        num_features=10, 
                        num_samples=200 
                    )
                    lime_progress.progress(100)
                    
                    # 5. 展示结果
                    st.write("### 🔥 文本热力图")
                    st.caption("颜色越红/深橙色，代表该词句越具有“原著神韵”；蓝色则代表与原著风格不符。")
                    components.html(exp.as_html(), height=600, scrolling=True)
                    
                    # 6. 提取关键词表
                    st.write("### 🏆 核心特征词")
                    top_features = exp.as_list()
                    # 只要正向特征
                    pos_feats = [f for f in top_features if f[1] > 0]
                    if pos_feats:
                        df_feats = pd.DataFrame(pos_feats, columns=["特征词", "原著感贡献度"])
                        st.dataframe(df_feats, use_container_width=True)
                    else:
                        st.write("未检测到显著的正向特征词。")
                        
                except Exception as e:
                    st.error(f"LIME 分析运行时出现错误: {e}")
                
                status.update(label="全流程分析已完成！", state="complete", expanded=False)

