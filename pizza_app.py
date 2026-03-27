import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pizza_processing import get_cleaned_data
import datetime

# --- 1. HÀM ĐỊNH DẠNG TIỀN VNĐ THÔNG MINH (DỄ ĐỌC) ---
def format_vnd_smart(amount):
    if amount >= 1_000_000_000: # Trên 1 Tỷ
        return f"{amount / 1_000_000_000:.2f} Tỷ VNĐ"
    elif amount >= 1_000_000:    # Trên 1 Triệu
        return f"{amount / 1_000_000:.1f} Triệu VNĐ"
    else:
        return f"{amount:,.0f} VNĐ"

# 2. Cấu hình trang
st.set_page_config(
    page_title="Pizza BI - Hệ Thống Quản Trị Thông Minh",
    layout="wide",
    page_icon="🍕",
    initial_sidebar_state="expanded"
)

# 3. GIAO DIỆN NÂNG CAO (CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f1f5f9; }
    .block-container { padding: 2rem 4rem !important; }
    [data-testid="stMetric"] {
        background: white; border-radius: 16px; padding: 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 6px solid #ef4444;
    }
    .stPlot { background-color: white !important; padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0; }
    .sidebar-footer { padding: 15px; background: #f8fafc; border-radius: 10px; margin-top: 20px; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# 4. QUẢN LÝ ĐĂNG NHẬP
if "logged_in" not in st.session_state: st.session_state.logged_in = False
def login(): st.session_state.logged_in = True
def logout(): st.session_state.logged_in = False

if not st.session_state.logged_in:
    col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
    with col_l2:
        st.markdown("<br><br><h1 style='text-align: center;'>Pizza BI Login</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            u = st.text_input("Tên đăng nhập")
            p = st.text_input("Mật khẩu", type="password")
            submit_button = st.form_submit_button("ĐĂNG NHẬP HỆ THỐNG")
            
            if submit_button:
                if u == "admin" and p == "123":
                    login(); st.rerun()
                else:
                    st.error("❌ Sai tên đăng nhập hoặc mật khẩu!")
else:
    @st.cache_data
    def load_data(): return get_cleaned_data()
    df = load_data()

   
    min_date_str = df['date'].min().strftime('%d/%m/%Y')
    max_date_str = df['date'].max().strftime('%d/%m/%Y')

    # SIDEBAR
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3595/3595455.png", width=80)
        st.title("PIZZA AI SUITE")
        st.info(f"📅 **Dữ liệu thực tế:** \n{min_date_str} - {max_date_str}")
        
        menu = st.radio("CHIẾN LƯỢC:", ["📈 KPI & Dự báo", "👥 Nhân sự", "📦 Vật tư & Kho", "🍕 Menu (BCG)", "🛒 Combo AI"])
        if st.button("🚪 Đăng xuất"): logout(); st.rerun()
        st.markdown(f"<div class='sidebar-footer'><b>SV thực hiện:</b> Thắng<br>Đồ án TN 2026<br><span style='color:red'>Ver 6.3 Smart VND</span></div>", unsafe_allow_html=True)

    # --- MODULE 0: KPI & DỰ BÁO DOANH THU ---
    if menu == "📈 KPI & Dự báo":
        st.title("📈 Phân tích KPI & Giải thích Mô hình AI")
        st.caption(f"📍 Phạm vi dữ liệu gốc: Năm 2015")
        
        df_ml = df.groupby('date').agg({'revenue': 'sum'}).reset_index().sort_values('date')
        df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
        df_ml['month'] = df_ml['date'].dt.month
        df_ml['lag_7d'] = df_ml['revenue'].shift(7).fillna(df_ml['revenue'].mean())
        
        X = df_ml[['day_of_week', 'month', 'lag_7d']]
        y = df_ml['revenue']
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

        # HIỂN THỊ METRICS DÙNG HÀM ĐỌC SỐ THÔNG MINH
        k1, k2, k3 = st.columns(3)
        k1.metric("Tổng Doanh thu", format_vnd_smart(df['revenue'].sum()))
        k2.metric("Doanh thu TB/Ngày", format_vnd_smart(df_ml['revenue'].mean()))
        k3.metric("Độ tin cậy AI (R2)", f"{r2_score(y, model.predict(X)):.2f}")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.subheader("📊 Dự báo Doanh thu 7 ngày tới")
            future_dates = [df_ml['date'].max() + datetime.timedelta(days=i) for i in range(1, 8)]
            future_X = pd.DataFrame({
                'day_of_week': [d.weekday() for d in future_dates],
                'month': [d.month for d in future_dates],
                'lag_7d': df_ml['revenue'].tail(7).values
            })
            preds = model.predict(future_X)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(x=future_dates, y=preds, marker='o', color='#ef4444', ax=ax)
            # Format trục Y để dễ đọc hơn trên biểu đồ
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e-6:.1f}M'))
            ax.set_ylabel("Triệu VNĐ")
            st.pyplot(fig)

        with col_b:
            st.subheader("🧠 Tại sao AI dự báo vậy?")
            importances = pd.DataFrame({'Yếu tố': X.columns, 'Độ ảnh hưởng': model.feature_importances_})
            fig_imp, ax_imp = plt.subplots()
            sns.barplot(data=importances.sort_values('Độ ảnh hưởng'), x='Độ ảnh hưởng', y='Yếu tố', palette='Reds_r')
            st.pyplot(fig_imp)

    # --- MODULE 1: NHÂN SỰ ---
    elif menu == "👥 Nhân sự":
        st.title("👥 Tối ưu hóa Nguồn lực & Nhân sự")
        st.caption(f"📍 Phân tích dựa trên lưu lượng khách năm 2015")
        
        num_days = df['date'].nunique()
        peak_hour = df.groupby('hour')['order_id'].nunique().idxmax()
        total_orders_at_peak = df.groupby('hour')['order_id'].nunique().max()
        avg_orders_per_hour_real = total_orders_at_peak / num_days
        suggested_staff = int(np.ceil(avg_orders_per_hour_real / 6)) + 1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🕒 Giờ Cao Điểm", f"{peak_hour}:00")
        c2.metric("📈 Đơn hàng TB/H", f"{avg_orders_per_hour_real:.1f}")
        c3.metric("⚡ Hiệu suất", "6 Đơn/H")
        c4.metric("👨‍🍳 Nhân sự cần", f"{suggested_staff} người")

        tab1, tab2 = st.tabs(["🌡️ Bản đồ nhiệt (Heatmap)", "📋 Lịch trình chi tiết"])
        with tab1:
            st.subheader("Mật độ đơn hàng trung bình mỗi giờ")
            heatmap_data = df.pivot_table(index='day_name', columns='hour', values='order_id', aggfunc='nunique').fillna(0)
            heatmap_avg = heatmap_data / num_days
            fig_h, ax_h = plt.subplots(figsize=(16, 7))
            sns.heatmap(heatmap_avg, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax_h)
            st.pyplot(fig_h)
        with tab2:
            st.subheader("📅 Công cụ điều phối ca trực")
            sel_day = st.selectbox("Chọn thứ trong tuần:", df['day_name'].unique())
            num_specific_days = df[df['day_name'] == sel_day]['date'].nunique()
            day_data = df[df['day_name'] == sel_day].groupby('hour')['order_id'].nunique()
            staff_list = []
            for hr in range(10, 24):
                avg_o = day_data.get(hr, 0) / num_specific_days
                staff = int(np.ceil(avg_o / 6)) + 1 if avg_o > 0 else 0
                staff_list.append({"Giờ": f"{hr}h", "Đơn dự kiến": round(avg_o, 1), "Nhân sự": staff})
            st.table(pd.DataFrame(staff_list))

    # --- MODULE 2: VẬT TƯ & KHO ---
    elif menu == "📦 Vật tư & Kho":
        st.title("📦 Quản trị Nguồn cung & Xuất báo cáo")
        ts_data = df.groupby(['date', 'hour']).agg({'order_id': 'nunique'}).reset_index()
        ts_data['d_week'] = ts_data['date'].dt.dayofweek
        ts_data['lag_1h'] = ts_data['order_id'].shift(1).fillna(0)
        X_inv = ts_data[['hour', 'd_week', 'lag_1h']]
        y_inv = ts_data['order_id']
        model_inv = RandomForestRegressor(n_estimators=50).fit(X_inv, y_inv)

        c_inv1, c_inv2 = st.columns([1, 2])
        with c_inv1:
            target_h = st.slider("Giờ dự báo", 10, 23, 18)
            buffer = st.slider("Biên an toàn (%)", 0, 50, 15)
            pred_val = model_inv.predict([[target_h, datetime.date.today().weekday(), 5]])[0] * (1 + buffer/100)
            st.metric("Dự báo Đơn hàng", f"{pred_val:.1f}")
        with c_inv2:
            st.subheader("📋 Danh sách vật tư dự kiến")
            inv_report = pd.DataFrame({
                "Nguyên liệu": ["Đế bánh (cái)", "Phô mai (kg)", "Sốt cà (lít)", "Hộp carton"],
                "Số lượng": [int(pred_val), pred_val*0.15, pred_val*0.1, int(pred_val)]
            })
            st.table(inv_report)
            st.download_button("📥 Tải Báo cáo Vật tư", inv_report.to_csv(index=False).encode('utf-8'), "bao_cao_kho.csv")

    # --- MODULE 3: MENU (BCG) ---
    elif menu == "🍕 Menu (BCG)":
        st.title("🍕 Phân tích Chiến lược Danh mục")
        p_data = df.groupby('pizza_name').agg({'revenue': 'sum', 'quantity': 'sum', 'price': 'mean'}).reset_index()
        sc = StandardScaler()
        p_scaled = sc.fit_transform(p_data[['quantity', 'price']])
        p_data['cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(p_scaled)
        
        fig_bcg, ax_bcg = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=p_data, x='quantity', y='price', hue='cluster', s=200, palette='viridis', ax=ax_bcg)
        ax_bcg.set_ylabel("Giá bán (VNĐ)")
        # Format trục Y sang Triệu/Nghìn cho sạch
        ax_bcg.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        st.pyplot(fig_bcg)

    # --- MODULE 4: COMBO AI ---
    elif menu == "🛒 Combo AI":
        st.title("🛒 Trình Thiết kế Combo & Gợi ý Thông minh")
        rules = pd.DataFrame() 
        basket = (df.groupby(['order_id', 'pizza_name'])['quantity']
                  .sum().unstack().reset_index().fillna(0).set_index('order_id'))
        basket_enc = basket.map(lambda x: 1 if x >= 1 else 0)

        col_set, col_rules = st.columns([1, 3])
        with col_set:
            sup = st.slider("Độ phổ biến", 0.001, 0.02, 0.005, format="%.3f")
            lift_val = st.slider("Sức mạnh liên kết", 1.0, 5.0, 1.2)
        with col_rules:
            f_sets = apriori(basket_enc, min_support=sup, use_colnames=True)
            if not f_sets.empty:
                rules = association_rules(f_sets, metric="lift", min_threshold=lift_val)
                if not rules.empty:
                    rules['Món đã mua'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['Món gợi ý'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    st.dataframe(rules[['Món đã mua', 'Món gợi ý', 'confidence', 'lift']].head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 Trình Mô phỏng Gợi ý")
        if not rules.empty:
            pick = st.selectbox("Khách chọn món:", sorted(df['pizza_name'].unique()))
            match = rules[rules['antecedents'].apply(lambda x: pick in x)]
            if not match.empty:
                best = match.sort_values('confidence', ascending=False).iloc[0]
                st.success(f"🤖 AI gợi ý: Mời khách mua thêm **{', '.join(list(best['consequents']))}**!")

    # FOOTER
    st.markdown("---")
    st.caption(f"Hệ thống Pizza BI v6.3 - © 2026 | Dữ liệu gốc 2015 đã được quy đổi sang VNĐ")