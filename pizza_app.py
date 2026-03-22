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

# 1. Cấu hình trang (Phải đặt ở đầu file)
st.set_page_config(
    page_title="Pizza BI - Hệ Thống Quản Trị Thông Minh",
    layout="wide",
    page_icon="🍕",
    initial_sidebar_state="expanded"
)

# 2. GIAO DIỆN NÂNG CAO (PREMIUM ENTERPRISE UI)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f1f5f9;
    }

    /* Tối ưu hóa Container chính */
    .block-container {
        padding: 3rem 5rem !important;
    }

    /* Thiết kế Thẻ (Cards) hiện đại cho Metrics */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 25px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08);
        border-color: #ef4444;
    }

    /* Hiệu ứng thanh màu trên thẻ Metric */
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 6px; height: 100%;
        background: linear-gradient(to bottom, #ef4444, #991b1b);
    }

    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #64748b !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Styling cao cấp */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        width: 320px !important;
    }

    .stRadio [data-testid="stWidgetLabel"] p {
        font-weight: 700;
        color: #1e293b;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }

    /* Tùy chỉnh các lựa chọn trong Menu Sidebar */
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] {
        gap: 10px;
    }

    div[data-testid="stSidebarUserContent"] .stRadio label {
        background: #f8fafc;
        border-radius: 10px;
        padding: 12px 16px !important;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
        cursor: pointer;
    }

    div[data-testid="stSidebarUserContent"] .stRadio label:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
    }

    div[data-testid="stSidebarUserContent"] .stRadio label[data-selected="true"] {
        background: #fef2f2 !important;
        border-color: #ef4444 !important;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
    }

    /* Tùy chỉnh Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e2e8f0;
        padding: 6px;
        border-radius: 12px;
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #475569;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #ef4444 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Tiêu đề chính h1 */
    h1 {
        color: #0f172a;
        font-weight: 800;
        font-size: 2.8rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 2rem !important;
    }

    /* Card bao quanh các Plot/Biểu đồ */
    .stPlot, .stTable {
        background-color: white !important;
        padding: 24px !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* Sidebar Footer */
    .sidebar-footer {
        padding: 20px;
        background: #f8fafc;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #e2e8f0;
    }

    /* Login Form Styling */
    .login-box {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: auto;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# QUẢN LÝ TRẠNG THÁI ĐĂNG NHẬP (SESSION STATE)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.session_state.logged_in = True

def logout():
    st.session_state.logged_in = False


# MÀN HÌNH ĐĂNG NHẬP

if not st.session_state.logged_in:
    col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
    with col_l2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3595/3595455.png", width=100)
        st.markdown("<h1 style='text-align: center; font-size: 2rem !important;'>Pizza BI Login</h1>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập", placeholder="Tên đăng nhập...")
            password = st.text_input("Mật khẩu", type="password", placeholder="Mật khẩu...")
            submit_login = st.form_submit_button("ĐĂNG NHẬP HỆ THỐNG")
            
            if submit_login:
                if username == "admin" and password == "123": # Tài khoản giả định để demo
                    login()
                    st.rerun()
                else:
                    st.error("❌ Sai tên đăng nhập hoặc mật khẩu!")
else:

    # LOAD DATA
   
    @st.cache_data
    def load_data():
        return get_cleaned_data()

    df = load_data()

   
    # SIDEBAR - BRANDING & NAVIGATION
   
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3595/3595455.png", width=90)
        st.title("PIZZA AI SUITE")
        st.markdown("Hệ thống Hỗ trợ Quyết định (DSS)")
        st.markdown("---")
        
        menu = st.radio(
            "MỤC TIÊU CHIẾN LƯỢC:", 
            [
                "📈 Quản trị KPI & Mục tiêu",
                "👥 Tối ưu Ca làm & Nhân sự", 
                "📦 Dự báo Nhu cầu & Vật tư", 
                "🍕 Chiến lược Menu (BCG Matrix)",
                "🛒 Khai phá Combo & Upselling"
            ]
        )
        
        st.markdown("---")
        
        # Status Cards trong Sidebar
        st.markdown("### 🛰️ Trạng thái Hệ thống")
        st.success("🟢 Cơ sở dữ liệu: Đã kết nối")
        st.info("🤖 Mô hình AI: Sẵn sàng")
        
        # Logout Button
        if st.button("🚪 Đăng xuất"):
            logout()
            st.rerun()

        st.markdown(f"""
            <div class='sidebar-footer'>
                <p style='margin:0; font-size: 0.8rem; color: #64748b; font-weight: 600;'>THÔNG TIN SINH VIÊN</p>
                <p style='margin:0; font-size: 0.9rem; color: #0f172a; font-weight: 700;'>Họ tên: Thắng</p>
                <p style='margin:0; font-size: 0.8rem; color: #64748b;'>Đồ án Tốt nghiệp 2026</p>
                <p style='margin-top:10px; font-size: 0.75rem; color: #ef4444; font-weight: 700;'>Ver 5.7 Enterprise</p>
            </div>
        """, unsafe_allow_html=True)

   
    # MODULE 0: KPI MANAGEMENT (NEW)
  
    if menu == "📈 Quản trị KPI & Mục tiêu":
        st.title("📈 Phân tích KPI & Dự báo Mục tiêu")
        st.markdown("Hệ thống đo lường hiệu quả kinh doanh so với mục tiêu và dự báo tăng trưởng.")

        # Tính toán các KPI cốt lõi
        total_revenue = df['revenue'].sum()
        total_orders = df['order_id'].nunique()
        aov = total_revenue / total_orders # Average Order Value
        target_revenue = 850000.0 # Mục tiêu giả định cho kỳ này
        completion_rate = (total_revenue / target_revenue) * 100

        # Hiển thị Metrics KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tổng Doanh thu", f"${total_revenue:,.0f}")
        k2.metric("Mục tiêu kỳ", f"${target_revenue:,.0f}")
        k3.metric("Tỷ lệ hoàn thành", f"{completion_rate:.1f}%", delta=f"{completion_rate-100:.1f}%")
        k4.metric("Giá trị Đơn TB (AOV)", f"${aov:.2f}")

        st.markdown("---")
        tab_k1, tab_k2 = st.tabs(["📊 Xu hướng Doanh thu", "🔮 Dự báo KPI Tương lai"])

        with tab_k1:
            st.subheader("Diễn biến Doanh thu theo thời gian")
            daily_rev = df.groupby('date')['revenue'].sum().reset_index()
            fig_rev, ax_rev = plt.subplots(figsize=(16, 6))
            sns.lineplot(data=daily_rev, x='date', y='revenue', color='#ef4444', linewidth=2.5, ax=ax_rev)
            ax_rev.fill_between(daily_rev['date'], daily_rev['revenue'], color='#ef4444', alpha=0.1)
            ax_rev.set_title("Biểu đồ tăng trưởng doanh thu hàng ngày", fontsize=14, fontweight='bold')
            st.pyplot(fig_rev)
            
            st.info(f"💡 **Phân tích:** Doanh thu trung bình mỗi ngày đạt **${daily_rev['revenue'].mean():,.2f}**. Để đạt mục tiêu, cần duy trì mức tăng trưởng ít nhất 5% mỗi tuần.")

        with tab_k2:
            st.subheader("🔮 Dự báo Doanh thu kỳ tới (Next Period Prediction)")
            
            # Huấn luyện mô hình dự báo doanh thu tổng hợp
            df_ml = df.groupby('date').agg({'revenue': 'sum'}).reset_index()
            df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
            df_ml['month'] = df_ml['date'].dt.month
            
            X_k = df_ml[['day_of_week', 'month']]
            y_k = df_ml['revenue']
            
            model_k = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_k, y_k)
            
            # Dự báo cho 7 ngày tới
            future_dates = [df_ml['date'].max() + datetime.timedelta(days=i) for i in range(1, 8)]
            future_X = pd.DataFrame({
                'day_of_week': [d.weekday() for d in future_dates],
                'month': [d.month for d in future_dates]
            })
            future_preds = model_k.predict(future_X)
            
            forecast_df = pd.DataFrame({'Ngày': future_dates, 'Dự báo Doanh thu ($)': future_preds})
            
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                fig_f, ax_f = plt.subplots(figsize=(10, 6))
                sns.barplot(data=forecast_df, x='Ngày', y='Dự báo Doanh thu ($)', palette='Reds_r', ax=ax_f)
                plt.xticks(rotation=45)
                st.pyplot(fig_f)
            
            with col_f2:
                total_forecast = future_preds.sum()
                st.markdown(f"""
                    <div style="background: white; padding: 25px; border-radius: 16px; border-left: 8px solid #ef4444; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                        <p style="margin:0; color:#64748b; font-weight:600;">TỔNG DỰ BÁO 7 NGÀY TỚI</p>
                        <h2 style="margin:0; color:#1e293b; font-size: 2.2rem;">${total_forecast:,.0f}</h2>
                        <p style="margin-top:10px; font-size: 0.85rem; color:#10b981;"><b>Khả thi cao:</b> Dựa trên dữ liệu lịch sử.</p>
                    </div>
                """, unsafe_allow_html=True)

  
    # MODULE 1: STAFFING OPTIMIZATION
   
    elif menu == "👥 Tối ưu Ca làm & Nhân sự":
        st.title("👥 Tối ưu hóa Nguồn lực")
        
        # KPIs trong Container chuyên nghiệp
        num_days = df['date'].nunique()
        peak_hour = df.groupby('hour')['order_id'].nunique().idxmax()
        total_orders_at_peak = df.groupby('hour')['order_id'].nunique().max()
        avg_orders_per_hour_real = total_orders_at_peak / num_days
        suggested_staff = int(np.ceil(avg_orders_per_hour_real / 6)) + 1

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("🕒 Giờ Cao Điểm", f"{peak_hour}:00")
        with c2: st.metric("📈 Đơn hàng TB/H", f"{avg_orders_per_hour_real:.1f}")
        with c3: st.metric("⚡ Hiệu suất TB", "6 Đơn/H")
        with c4: st.metric("👨‍🍳 Nhân sự đề xuất", f"{suggested_staff} người")

        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🌡️ Bản đồ nhiệt (Heatmap)", "📋 Lịch trình chi tiết"])
        
        with tab1:
            st.subheader("Phân tích mật độ đơn hàng theo thời gian")
            heatmap_data = df.pivot_table(index='day_name', columns='hour', values='order_id', aggfunc='nunique').fillna(0)
            heatmap_avg = heatmap_data / num_days
            
            fig, ax = plt.subplots(figsize=(16, 7))
            sns.heatmap(heatmap_avg, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax, linewidths=1, cbar_kws={'label': 'Đơn hàng trung bình/ngày'})
            ax.set_title("Mật độ đơn hàng trung bình mỗi giờ", fontsize=14, fontweight='bold', pad=20)
            st.pyplot(fig)

        with tab2:
            st.subheader("📅 Công cụ điều phối ca trực")
            sel_day = st.selectbox("Chọn ngày cần dự tính nhân sự:", df['day_name'].unique())
            num_specific_days = df[df['day_name'] == sel_day]['date'].nunique()
            day_data = df[df['day_name'] == sel_day].groupby('hour')['order_id'].nunique()
            
            staff_list = []
            for hr in range(10, 24):
                avg_orders = day_data.get(hr, 0) / num_specific_days
                staff = int(np.ceil(avg_orders / 6)) + 1 if avg_orders > 0 else 0
                staff_list.append({
                    "Khung Giờ": f"{hr}h - {hr+1}h", 
                    "Đơn dự kiến": round(avg_orders, 1), 
                    "Nhân sự cần thiết": staff,
                    "Trạng thái": "🔥 Cao điểm" if avg_orders > 5 else "🟢 Ổn định"
                })
            st.table(pd.DataFrame(staff_list).set_index("Khung Giờ"))

    
    # MODULE 2: INVENTORY FORECASTING
    
    elif menu == "📦 Dự báo Nhu cầu & Vật tư":
        st.title("📦 Quản trị Nguồn cung AI")
        
        ts_data = df.groupby(['date', 'hour']).agg({'order_id': 'nunique'}).reset_index()
        ts_data.columns = ['date', 'hour', 'y']
        ts_data['d_week'] = ts_data['date'].dt.dayofweek
        ts_data['month'] = ts_data['date'].dt.month
        X = ts_data[['hour', 'd_week', 'month']]
        y = ts_data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

        col_conf, col_pred = st.columns([1, 2])
        
        with col_conf:
            st.markdown("### ⚙️ Tham số Dự báo")
            target_date = st.date_input("Chọn ngày:", datetime.date.today() + datetime.timedelta(days=1))
            target_hour = st.select_slider("Khung giờ:", options=list(range(10, 24)), value=18)
            safety_buf = st.slider("Biên an toàn (%)", 0, 50, 15)
            
        with col_pred:
            input_v = pd.DataFrame([[target_hour, target_date.weekday(), target_date.month]], columns=['hour', 'd_week', 'month'])
            raw_p = rf_model.predict(input_v)[0]
            final_p = raw_p * (1 + safety_buf/100)
            
            st.markdown(f"""
                <div style="background: white; padding: 30px; border-radius: 16px; border-left: 8px solid #ef4444; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                    <p style="margin:0; color:#64748b; font-weight:600; text-transform:uppercase; font-size:0.8rem;">DỰ BÁO NHU CẦU THEO GIỜ</p>
                    <h2 style="margin:0; color:#0f172a; font-size: 3.5rem; font-weight:800;">{final_p:.1f} <span style="font-size:1.2rem; color:#94a3b8; font-weight:400;">đơn hàng</span></h2>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🥗 Định mức Vật tư Khuyến nghị")
            m1, m2, m3 = st.columns(3)
            m1.metric("🍞 Bột đế (kg)", f"{final_p * 0.25:.2f}")
            m2.metric("🧀 Phô mai (kg)", f"{final_p * 0.18:.2f}")
            m3.metric("🍅 Sốt cà (lít)", f"{final_p * 0.12:.2f}")

    
    # MODULE 3: BCG MATRIX
    
    elif menu == "🍕 Chiến lược Menu (BCG Matrix)":
        st.title("🍕 Chiến lược Menu AI")
        
        p_data = df.groupby('pizza_name').agg({'revenue': 'sum', 'quantity': 'sum', 'price': 'mean'}).reset_index()
        sc = StandardScaler()
        p_scaled = sc.fit_transform(p_data[['quantity', 'price']])
        km = KMeans(n_clusters=4, random_state=42, n_init=15).fit(p_scaled)
        p_data['cluster'] = km.labels_
        
        col_plot, col_strat = st.columns([3, 2])
        
        with col_plot:
            st.subheader("📍 Ma trận Phân cụm Sản phẩm")
            fig, ax = plt.subplots(figsize=(10, 8))
            q_m, p_m = p_data['quantity'].mean(), p_data['price'].mean()
            sns.scatterplot(data=p_data, x='quantity', y='price', hue='cluster', palette='coolwarm', s=300, ax=ax, edgecolor='black', alpha=0.9)
            plt.axvline(q_m, color='#94a3b8', linestyle='--', alpha=0.5)
            plt.axhline(p_m, color='#94a3b8', linestyle='--', alpha=0.5)
            for i, row in p_data.nlargest(6, 'quantity').iterrows():
                ax.text(row['quantity']+10, row['price'], row['pizza_name'], fontsize=9, fontweight='bold')
            st.pyplot(fig)

        with col_strat:
            st.subheader("💡 Hành động Chiến lược")
            for cl in range(4):
                sub = p_data[p_data['cluster'] == cl]
                mq, mp = sub['quantity'].mean(), sub['price'].mean()
                with st.expander(f"Phân tích Nhóm {cl} ({len(sub)} món)"):
                    if mq >= q_m and mp >= p_m: st.success("🌟 STARS: Món chủ lực. Duy trì và đẩy mạnh.")
                    elif mq >= q_m: st.info("🐄 CASH COWS: Doanh thu ổn định. Tối ưu chi phí.")
                    elif mp >= p_m: st.warning("❓ QUESTION MARKS: Giá cao, ít khách. Cần Marketing.")
                    else: st.error("📉 DOGS: Bán chậm. Khuyến nghị loại bỏ.")
                    st.write("Top món: " + ", ".join(sub.nlargest(3, 'quantity')['pizza_name'].tolist()))

   
    # MODULE 4: ADVANCED APRIORI
  
    elif menu == "🛒 Khai phá Combo & Upselling":
        st.title("🛒 Trình Thiết kế Combo")
        
        basket = (df.groupby(['order_id', 'pizza_name'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('order_id'))
        basket_encoded = basket.map(lambda x: 1 if x >= 1 else 0)

        col_set, col_rules = st.columns([1, 3])
        with col_set:
            st.markdown("### 🧠 Cấu hình AI")
            sup = st.slider("Độ phổ biến (Support)", 0.001, 0.015, 0.005, format="%.3f")
            lift = st.slider("Sức mạnh liên kết (Lift)", 1.0, 5.0, 1.2)
        
        with col_rules:
            f_sets = apriori(basket_encoded, min_support=sup, use_colnames=True)
            if not f_sets.empty:
                rules = association_rules(f_sets, metric="lift", min_threshold=lift)
                if not rules.empty:
                    rules['A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    st.subheader("🔥 Top Quy luật Tiềm năng")
                    st.dataframe(rules[['A', 'B', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10).style.background_gradient(cmap='YlOrRd', subset=['lift']), use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("🤖 Trình Mô phỏng Gợi ý")
                    target = st.selectbox("Khách chọn món:", sorted(df['pizza_name'].unique()))
                    m = rules[rules['antecedents'].apply(lambda x: target in x)]
                    if not m.empty:
                        top = m.sort_values('confidence', ascending=False).iloc[0]
                        st.success(f"**💡 GỢI Ý:** Mời khách dùng thêm **{top['B']}**! (Tỷ lệ chốt đơn: {top['confidence']*100:.1f}%)")

    # FOOTER TRANG
    st.markdown("---")
    st.caption("Dữ liệu được xử lý bởi Hệ thống Enterprise Pizza BI v5.7 - © 2026")