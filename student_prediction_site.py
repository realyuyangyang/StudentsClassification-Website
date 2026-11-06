# -*- coding: utf-8 -*-
"""学生阅读障碍辅助筛查研究展示网站：10算法预测 + 单个学生SHAP解释。

本程序不会训练模型。模型由已有10算法训练脚本导出；本网站只加载可信的joblib模型包。
SHAP使用全部学生（默认642名）作为背景，解释当前选中模型的P(阅读障碍)输出。
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="阅读水平智能分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SITE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = Path(os.environ.get("DYSLEXIA_MODEL_DIR", str(SITE_DIR))).expanduser()
MANIFEST_FILE = DEPLOY_DIR / "manifest.json"
SHAP_PERMUTATIONS = int(os.environ.get("DYSLEXIA_SHAP_PERMUTATIONS", "5"))
SHAP_BATCH_SIZE = int(os.environ.get("DYSLEXIA_SHAP_BATCH_SIZE", "256"))


def configure_chinese_matplotlib():
    """在Streamlit Cloud/Colab环境中尽量启用中文字体。"""
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ]
        font_name = None
        for font_path in candidates:
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                    font_name = fm.FontProperties(fname=font_path).get_name()
                    break
                except Exception:
                    continue
        if font_name:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        else:
            matplotlib.rcParams["font.sans-serif"] = [
                "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"
            ]
        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["figure.figsize"] = (8, 6)
        matplotlib.rcParams["figure.dpi"] = 300
        matplotlib.rcParams["savefig.dpi"] = 300
        matplotlib.rcParams["font.size"] = 9
        matplotlib.rcParams["axes.titlesize"] = 11
        matplotlib.rcParams["axes.labelsize"] = 10
        matplotlib.rcParams["xtick.labelsize"] = 8
        matplotlib.rcParams["ytick.labelsize"] = 8
        matplotlib.rcParams["legend.fontsize"] = 8
    except Exception:
        pass


configure_chinese_matplotlib()


@st.cache_data(show_spinner=False)
def load_manifest(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cv_table(path: str, mtime_ns: int):
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in (
        "ROC_AUC_Mean", "ROC_AUC_SD", "AP_Mean", "AP_SD",
        "MCC_Mean", "阅读障碍_召回率_Mean",
    ):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource(show_spinner="正在加载模型...")
def load_model_bundle(path: str, mtime_ns: int):
    # joblib/pickle仅加载本项目自行生成、可信来源的文件。
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_shap_background(path: str, feature_names: tuple, expected_rows: int, mtime_ns: int):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "没有找到 shap_background.csv。请先在Colab运行部署包中的 "
            "prepare_shap_background.py。"
        )
    background = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in feature_names if c not in background.columns]
    extra = [c for c in background.columns if c not in feature_names]
    if missing or extra:
        raise ValueError(f"SHAP背景特征列与模型不一致。缺少={missing}；多余={extra}")
    background = background.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    if expected_rows and len(background) != int(expected_rows):
        raise ValueError(
            f"SHAP背景数据行数为{len(background)}，manifest要求{expected_rows}。"
            "请重新生成背景数据，确保使用全部学生。"
        )
    if len(background) < 2:
        raise ValueError("SHAP背景数据至少需要2名学生。")
    arr = background.to_numpy(dtype=float)
    if np.isinf(arr).any():
        raise ValueError("SHAP背景数据包含正负无穷值，请检查数据。")
    return background


def _get_bundle_parts(bundle, values: pd.DataFrame):
    if not isinstance(bundle, dict):
        raise TypeError("模型文件不是预期的预处理器+分类器模型包。")
    feature_names = list(bundle["feature_names"])
    frame = values.loc[:, feature_names].copy()
    matrix = bundle["imputer"].transform(frame)
    scaler = bundle.get("scaler")
    if scaler is not None:
        matrix = scaler.transform(matrix)
    selector = bundle.get("selector")
    if selector is not None:
        matrix = selector.transform(matrix)
    classifier = bundle["classifier"]
    return matrix, classifier


def positive_probabilities(bundle, values: pd.DataFrame):
    matrix, classifier = _get_bundle_parts(bundle, values)
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"模型不包含阅读障碍：{classes}")
    probabilities = np.asarray(classifier.predict_proba(matrix), dtype=float)
    return probabilities[:, classes.index(1)]


def positive_probability(bundle, values: pd.DataFrame) -> float:
    return float(positive_probabilities(bundle, values)[0])


@st.cache_data(show_spinner=False, max_entries=50)
def calculate_single_student_shap(
    model_path: str,
    model_mtime_ns: int,
    background_path: str,
    background_mtime_ns: int,
    feature_names: tuple,
    input_values: tuple,
    background_rows: int,
    permutations: int,
    batch_size: int,
):
    """用全部背景学生计算当前模型对P(阅读障碍)的Permutation SHAP。"""
    import shap

    bundle = joblib.load(model_path)
    features = list(feature_names)
    background = pd.read_csv(background_path, encoding="utf-8-sig")
    background = background.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if len(background) != int(background_rows):
        raise ValueError(f"背景样本数变化：预期{background_rows}，实际{len(background)}。")

    def model_function(array):
        arr = np.asarray(array, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        frame = pd.DataFrame(arr, columns=features)
        return positive_probabilities(bundle, frame)

    # max_samples明确设为全部背景行数，防止SHAP默认只抽取100行。
    masker = shap.maskers.Independent(
        background,
        max_samples=len(background),
    )
    explainer = shap.PermutationExplainer(
        model_function,
        masker,
        feature_names=features,
        seed=42,
    )
    row = np.asarray(input_values, dtype=float).reshape(1, -1)
    max_evals = (2 * len(features) + 1) * max(1, int(permutations))
    explanation = explainer(
        row,
        max_evals=max_evals,
        batch_size=max(1, int(batch_size)),
        silent=True,
    )

    shap_array = np.asarray(explanation.values, dtype=float)
    if shap_array.ndim == 3 and shap_array.shape[-1] == 1:
        shap_array = shap_array[..., 0]
    if shap_array.ndim == 2:
        shap_array = shap_array[0]
    if shap_array.shape[0] != len(features):
        raise ValueError(
            f"SHAP输出维度异常：{shap_array.shape}；特征数={len(features)}"
        )

    base_arr = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    base_value = float(base_arr[0])
    prediction = float(model_function(row)[0])
    reconstructed = float(base_value + shap_array.sum())
    return shap_array.tolist(), base_value, prediction, reconstructed, len(background)


def render_shap_plots(feature_names, input_values, shap_values, base_value):
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(
        values=np.asarray(shap_values, dtype=float),
        base_values=float(base_value),
        data=np.asarray(input_values, dtype=float),
        feature_names=list(feature_names),
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**SHAP Waterfall：单个学生贡献路径**")
        try:
            shap.plots.waterfall(
                explanation,
                max_display=len(feature_names),
                show=False,
            )
            fig = plt.gcf()
            fig.set_size_inches(8, 6, forward=True)
            fig.tight_layout(pad=2.0)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Waterfall图生成失败：{type(exc).__name__}: {exc}")

    with right:
        st.markdown("**SHAP贡献排序：绝对贡献大小**")
        try:
            vals = np.asarray(shap_values, dtype=float)
            order = np.argsort(np.abs(vals))
            names = np.asarray(feature_names, dtype=object)[order]
            ordered_vals = vals[order]
            colors = ["#D97706" if v > 0 else "#2563EB" for v in ordered_vals]
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.barh(names, ordered_vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("对阅读障碍预测分数的SHAP贡献")
            ax.set_title("单个学生14项特征贡献")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout(pad=2.0)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"SHAP贡献排序图生成失败：{type(exc).__name__}: {exc}")


st.title("阅读水平智能分析")
st.caption(
    "默认调用交叉验证选出的最佳算法；可手动切换10种算法，并查看当前模型对单个学生的SHAP解释。"
)

if not MANIFEST_FILE.exists():
    st.error(
        "未找到manifest.json。请将本网站脚本放在模型‘网站部署’目录中，"
        "与manifest.json、models/和shap_background.csv同级。"
    )
    st.stop()

try:
    meta = load_manifest(str(MANIFEST_FILE))
    ranking_path = DEPLOY_DIR / "cv_ranking.csv"
    ranking = load_cv_table(str(ranking_path), ranking_path.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"部署文件加载失败：{type(exc).__name__}: {exc}")
    st.stop()

model_files = meta.get("models", {})
model_order = meta.get("model_order", [])
available_names = [
    name for name in model_order
    if name in model_files and (DEPLOY_DIR / model_files[name]).is_file()
]
best = meta.get("best_model")
if not available_names:
    st.error("没有可加载的模型文件，请检查models/目录。")
    st.stop()
if best not in available_names:
    st.warning("交叉验证最佳模型文件不存在，暂时使用第一个可用模型。")
    best = available_names[0]

st.sidebar.header("模型选择")
selected_model = st.sidebar.selectbox(
    "当前预测算法",
    options=available_names,
    index=available_names.index(best),
    help="默认按5个随机种子的OOF平均ROC-AUC选择；其余模型可手动切换。",
)
st.sidebar.success(f"交叉验证最佳：{best}")
st.sidebar.caption(
    f"选择规则：{meta.get('selection_rule', '五个随机种子的OOF平均ROC-AUC，AP和MCC作平局判定')}"
)
st.sidebar.caption(
    f"主实验：{meta.get('n_seeds', 5)}个随机种子 × {meta.get('n_folds', 5)}折；"
    f"当前可用模型：{len(available_names)}/{len(model_order)}"
)

features = list(meta["feature_names"])
defaults = meta.get("feature_defaults", {})
CHAR_NAMES = [
    "声母", "韵母", "声调", "整字形", "结构",
    "部件", "笔画", "语义相似度", "笔顺轨迹"
]
TIME_NAMES = [
    "总体启动时长均值",
    "总体书写时长均值",
    "总体停顿时长均值"
]

prediction_tab, shap_tab, comparison_tab, method_tab = st.tabs(
    ["单个学生预测", "单个学生SHAP解释", "十算法性能与预测比较", "方法和解释"]
)

with prediction_tab:
    st.subheader("输入学生级特征")
    st.caption(
        "9项汉字正确率输入0–1的小数；三个时间指标的单位必须与训练数据一致。"
        "不需要输入姓名、学校、班级或学生ID。"
    )
    with st.form("student_features", clear_on_submit=False):
        col1, col2 = st.columns(2)
        entries = {}

        if "年级" in features:
            grades = [3, 4, 5, 6]
            try:
                grade_default = int(round(float(defaults.get("年级", 4))))
            except Exception:
                grade_default = 4
            if grade_default not in grades:
                grade_default = 4
            entries["年级"] = float(col1.selectbox(
                "年级", grades, index=grades.index(grade_default)
            ))

        if "性别" in features:
            try:
                gender_default = int(round(float(defaults.get("性别", 0))))
            except Exception:
                gender_default = 0
            entries["性别"] = float(col2.selectbox(
                "性别",
                options=[0, 1],
                index=int(gender_default == 1),
                format_func=lambda g: "男" if g == 0 else "女",
            ))

        st.markdown("**汉字正确率（0–1）**")
        col_a, col_b, col_c = st.columns(3)
        for j, name in enumerate(CHAR_NAMES):
            if name not in features:
                continue
            try:
                initial = float(defaults.get(name, 0.5))
            except Exception:
                initial = 0.5
            initial = max(0.0, min(1.0, initial))
            column = (col_a, col_b, col_c)[j % 3]
            entries[name] = float(column.number_input(
                name,
                min_value=0.0,
                max_value=1.0,
                value=round(initial, 4),
                step=0.01,
                format="%.4f",
                key=f"rate_{name}",
            ))

        st.markdown("**时间特征（单位与训练数据相同）**")
        time_cols = st.columns(3)
        for j, name in enumerate(TIME_NAMES):
            if name not in features:
                continue
            try:
                initial = max(0.0, float(defaults.get(name, 0.0)))
            except Exception:
                initial = 0.0
            entries[name] = float(time_cols[j].number_input(
                name,
                min_value=0.0,
                value=initial,
                step=10.0,
                format="%.2f",
                key=f"time_{name}",
            ))

        other = [name for name in features if name not in entries]
        if other:
            st.markdown("**其他模型特征**")
            for name in other:
                entries[name] = float(st.number_input(
                    name,
                    value=float(defaults.get(name, 0.0)),
                    key=f"other_{name}",
                ))

        compare_all = st.checkbox(
            "同时运行所有可用模型，进行横向预测比较",
            value=True,
        )
        submitted = st.form_submit_button(
            "执行预测",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        frame = pd.DataFrame(
            [{name: entries[name] for name in features}],
            columns=features,
        )
        st.session_state["last_inputs"] = frame
        st.session_state["compare_all"] = compare_all
        st.session_state.pop("last_shap_result", None)

    if "last_inputs" in st.session_state:
        frame = st.session_state["last_inputs"]
        names_to_predict = (
            available_names if st.session_state.get("compare_all", True)
            else [selected_model]
        )
        if selected_model not in names_to_predict:
            names_to_predict.append(selected_model)

        results = []
        for name in names_to_predict:
            try:
                model_path = DEPLOY_DIR / model_files[name]
                bundle = load_model_bundle(
                    str(model_path), model_path.stat().st_mtime_ns
                )
                score = positive_probability(bundle, frame)
                results.append({
                    "算法": name,
                    "模型分数": score,##模型分数
                    "预测类别": (
                        "阅读障碍"
                        if score >= float(meta.get("threshold", 0.5))
                        else "普通学生"
                    ),
                    "阈值": float(meta.get("threshold", 0.5)),
                })
            except Exception as exc:
                st.warning(
                    f"{name}预测失败，已跳过：{type(exc).__name__}: {exc}"
                )

        if results:
            result_df = pd.DataFrame(results)
            selected = result_df[result_df["算法"] == selected_model]
            if not selected.empty:
                score = float(selected.iloc[0]["模型分数"])
                a, b, c = st.columns(3)
                a.metric("当前预测算法", selected_model)
                b.metric("模型分数", f"{score:.3f}")
                c.metric(
                    "模型判定",
                    "阅读障碍"
                    if score >= float(meta.get("threshold", 0.5))
                    else "普通学生",
                )
                st.progress(min(1.0, max(0.0, score)))

            st.markdown("**同一输入的算法预测比较**")
            st.dataframe(
                result_df.sort_values(
                    "模型分数", ascending=False
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.bar_chart(
                result_df.set_index("算法")["模型分数"]
            )
            st.download_button(
                "下载本次预测对比CSV",
                result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="10模型_单样本预测对比.csv",
                mime="text/csv",
            )
            st.info(
                " "
                "我们希望通过科学的探索，让每一份阅读表现都能被清晰地看见，让每一个孩子都能获得适合自己的帮助与鼓励。"
            )

with shap_tab:
    st.subheader("单个学生的SHAP解释")
    st.caption(
        "解释当前侧边栏所选算法对 阅读障碍 预测分数的贡献。"
        "正SHAP值提高模型分数，负SHAP值降低模型分数；不代表因果关系。"
    )

    if "last_inputs" not in st.session_state:
        st.info("请先进入‘单个学生预测’，输入14项特征并执行预测。")
    else:
        background_name = meta.get(
            "shap_background_file", "shap_background.csv"
        )
        background_path = DEPLOY_DIR / background_name

        if not background_path.exists():
            st.error(
                "未找到SHAP背景文件。请在Colab运行部署包中的 "
                "prepare_shap_background.py，生成全部学生的shap_background.csv，"
                "再更新网站部署目录。"
            )
        else:
            background_rows = int(
                meta.get("shap_background_rows", meta.get("n_students", 0))
            )
            st.write(
                f"当前算法：**{selected_model}**　|　"
                f"SHAP背景样本：**{background_rows}名学生**　|　"
                f"特征数：**{len(features)}**"
            )
            st.caption(
                f"Permutation SHAP；每次解释使用{SHAP_PERMUTATIONS}组排列；"
                "背景数据为全部训练学生，不包含标签和身份字段。"
            )
            try:
                checked_background = load_shap_background(
                    str(background_path),
                    tuple(features),
                    background_rows,
                    background_path.stat().st_mtime_ns,
                )
                st.caption(
                    f"已校验背景数据：{len(checked_background)}行 × "
                    f"{checked_background.shape[1]}列；本次解释将使用全部背景行。"
                )
            except Exception as exc:
                st.error(f"SHAP背景数据检查失败：{type(exc).__name__}: {exc}")
                st.stop()

            if st.button(
                f"计算当前学生的 {selected_model} SHAP解释",
                type="primary",
                use_container_width=True,
            ):
                try:
                    model_path = DEPLOY_DIR / model_files[selected_model]
                    input_frame = st.session_state["last_inputs"]
                    input_values = tuple(
                        float(input_frame.iloc[0][name])
                        for name in features
                    )
                    with st.spinner(
                        f"正在使用全部{background_rows}名学生作为背景计算SHAP，"
                        "首次计算可能需要一些时间..."
                    ):
                        (
                            shap_values,
                            base_value,
                            prediction,
                            reconstructed,
                            used_rows,
                        ) = calculate_single_student_shap(
                            str(model_path),
                            model_path.stat().st_mtime_ns,
                            str(background_path),
                            background_path.stat().st_mtime_ns,
                            tuple(features),
                            input_values,
                            background_rows,
                            SHAP_PERMUTATIONS,
                            SHAP_BATCH_SIZE,
                        )
                    st.session_state["last_shap_result"] = {
                        "model": selected_model,
                        "input_values": input_values,
                        "shap_values": shap_values,
                        "base_value": base_value,
                        "prediction": prediction,
                        "reconstructed": reconstructed,
                        "background_rows": used_rows,
                    }
                except Exception as exc:
                    st.error(
                        f"SHAP计算失败：{type(exc).__name__}: {exc}"
                    )

            result = st.session_state.get("last_shap_result")
            if result and result.get("model") == selected_model:
                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "基准输出（Base value）",
                    f"{result['base_value']:.4f}",
                )
                m2.metric(
                    "模型预测分数",
                    f"{result['prediction']:.4f}",
                )
                m3.metric(
                    "Base + SHAP之和",
                    f"{result['reconstructed']:.4f}",
                )
                st.caption(
                    f"加和校验差值："
                    f"{abs(result['prediction'] - result['reconstructed']):.6f}；"
                    "Permutation SHAP为近似估计，可能存在微小数值误差。"
                )

                render_shap_plots(
                    features,
                    result["input_values"],
                    result["shap_values"],
                    result["base_value"],
                )

                contribution = pd.DataFrame({
                    "特征": features,
                    "输入值": result["input_values"],
                    "SHAP值": result["shap_values"],
                })
                contribution["绝对贡献"] = contribution["SHAP值"].abs()
                contribution["方向"] = np.where(
                    contribution["SHAP值"] > 0,
                    "提高模型分数",
                    np.where(
                        contribution["SHAP值"] < 0,
                        "降低模型分数",
                        "影响接近0",
                    ),
                )
                contribution = contribution.sort_values(
                    "绝对贡献", ascending=False
                ).reset_index(drop=True)
                contribution.insert(
                    0, "排名", np.arange(1, len(contribution) + 1)
                )

                st.markdown("**14项特征的个体SHAP贡献明细**")
                st.dataframe(
                    contribution,
                    hide_index=True,
                    use_container_width=True,
                )
                st.download_button(
                    "下载个体SHAP贡献CSV",
                    contribution.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_model}_单个学生_SHAP贡献.csv",
                    mime="text/csv",
                )
            elif result:
                st.info(
                    f"当前缓存解释来自{result.get('model')}。当前已切换为"
                    f"{selected_model}，请重新计算该算法的SHAP解释。"
                )

with comparison_tab:
    st.subheader("十算法交叉验证性能排名")
    st.caption(
        "每个随机种子执行一次分层五折CV；每个种子汇总一组OOF指标，"
        "再计算五次Mean ± SD。"
    )
    st.dataframe(ranking, hide_index=True, use_container_width=True)
    if "ROC_AUC_Mean" in ranking and "算法" in ranking:
        st.subheader("ROC-AUC比较")
        st.bar_chart(
            ranking.set_index("算法")["ROC_AUC_Mean"].sort_values(
                ascending=False
            )
        )
    if "AP_Mean" in ranking and "算法" in ranking:
        st.subheader("PR-AUC / Average Precision比较")
        st.bar_chart(
            ranking.set_index("算法")["AP_Mean"].sort_values(
                ascending=False
            )
        )
    st.download_button(
        "下载CV模型排名表",
        ranking.to_csv(index=False).encode("utf-8-sig"),
        file_name="CV_模型排名.csv",
        mime="text/csv",
    )
    st.caption(
        "最佳算法是在当前交叉验证结果中选择，最佳模型的CV成绩存在模型选择偏倚；"
        "正式科研结论需要独立外部验证。"
    )

with method_tab:
    st.markdown(
        """
        **模型与解释方法**

        - **算法**：Logistic Regression、SVM-RBF、Random Forest、XGBoost、CatBoost、
          LightGBM、EBM、Gradient Boosting、Gaussian Naive Bayes、MLPClassifier。
        - **模型选择**：默认使用五个随机种子的OOF平均ROC-AUC最高算法；其余模型可手动切换。
        - **个体解释**：使用Permutation SHAP解释当前算法的阅读障碍预测分数。
        - **背景数据**：使用全部训练学生的14项特征；不包含Label、姓名、学校、班级、StudentKey或学生ID。
        - **解释含义**：正SHAP值表示该特征使模型的阅读障碍输出高于基准输出；负SHAP值表示使输出低于基准输出。

        """
    )
    st.caption(f"模型特征顺序：{', '.join(features)}")
