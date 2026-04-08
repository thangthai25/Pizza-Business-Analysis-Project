import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from mlxtend.frequent_patterns import apriori, association_rules
from pizza_processing import get_cleaned_data 

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="PIZZA AI SUITE | Ver 7.1", layout="wide", page_icon="🍕")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    [data-testid="stMetric"] {
        background: white; border-radius: 12px; padding: 15px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 5px solid #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TẢI DỮ LIỆU ---
@st.cache_data
def load_all_data():
    df = get_cleaned_data()
    if df is not None:
        df_ml = df.groupby(['date', 'hour']).agg({'revenue': 'sum'}).reset_index()
        df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
        df_ml['month'] = df_ml['date'].dt.month
        df_ml['is_weekend'] = df_ml['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_ml['lag_1h'] = df_ml['revenue'].shift(1).fillna(df_ml['revenue'].median())
        return df, df_ml
    return None, None

df, df_ml = load_all_data()

if df is not None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🍕 PIZZA AI SUITE")
        mode = st.radio("PHÂN HỆ HỆ THỐNG:", 
                        ["🔮 Dự báo & What-if", "🛒 Chiến lược Combo AI", "👥 Sắp xếp Nhân sự"])
        st.divider()
        st.markdown(f"**SV thực hiện:** Thắng<br>Đồ án TN 2026", unsafe_allow_html=True)

    # --- PHÂN HỆ 1: DỰ BÁO ---
    if mode == "🔮 Dự báo & What-if":
        st.title("🔮 Dự báo Doanh thu & Kịch bản Giả định")
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1h']
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df_ml[features], df_ml['revenue'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Độ tin cậy AI", f"{r2_score(df_ml['revenue'], model.predict(df_ml[features])):.2f}")
        m2.metric("Sai số TB (MAE)", f"{mean_absolute_error(df_ml['revenue'], model.predict(df_ml[features])):,.0f} VNĐ")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📊 Xu hướng doanh thu 24h tới")
            future = pd.DataFrame({'hour': range(24), 'day_of_week': 5, 'month': 12, 'is_weekend': 1, 'lag_1h': [4000000]*24})
            preds = model.predict(future)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(range(24), preds, marker='o', color='red')
            st.pyplot(fig)
        with col2:
            st.subheader("🧪 Chạy thử kịch bản")
            h = st.slider("Chọn giờ", 10, 23, 19)
            if st.button("Dự báo ngay"):
                res = model.predict([[h, 5, 12, 1, 4000000]])[0]
                st.write(f"Dự kiến: **{res:,.0f} VNĐ**")

    # --- PHÂN HỆ 2: COMBO AI ---
    elif mode == "🛒 Chiến lược Combo AI":
        st.title("🛒 Gợi ý Combo tối ưu lợi nhuận")
        basket = df.groupby(['order_id', 'pizza_name'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('order_id')
        basket_sets = basket.map(lambda x: 1 if x >= 1 else 0)
        frequent = apriori(basket_sets, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent, metric="lift", min_threshold=1.2)
        st.dataframe(rules[['antecedents', 'consequents', 'confidence', 'lift']].head(10))

    # --- PHÂN HỆ 3: NHÂN SỰ (PHẦN BẠN CẦN) ---
    elif mode == "👥 Sắp xếp Nhân sự":
        st.title("👥 Quản trị & Sắp xếp Lịch trực Nhân sự")
        
        # Tính toán chỉ số tổng quát
        num_days = df['date'].nunique()
        peak_hour = df.groupby('hour')['order_id'].nunique().idxmax()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🕒 Giờ Cao Điểm Hệ Thống", f"{peak_hour}:00")
        c2.metric("📈 Công suất chuẩn", "6 Đơn hàng/Người/Giờ")
        c3.metric("📅 Tổng ngày phân tích", f"{num_days} ngày")

        tab1, tab2 = st.tabs(["🌡️ Phân tích Mật độ (Heatmap)", "📋 Lịch trình chi tiết (Staff Schedule)"])
        
        with tab1:
            st.subheader("Lưu lượng đơn hàng trung bình mỗi giờ")
            # pivot_table để vẽ heatmap
            heatmap_data = df.pivot_table(index='day_name', columns='hour', values='order_id', aggfunc='nunique').fillna(0) / num_days
            fig_h, ax_h = plt.subplots(figsize=(16, 7))
            sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax_h)
            st.pyplot(fig_h)

        with tab2:
            st.subheader("📅 Công cụ lập lịch trực theo Thứ")
            sel_day = st.selectbox("Chọn ngày trong tuần để xếp lịch:", df['day_name'].unique())
            
            # Lọc dữ liệu theo thứ đã chọn
            day_data = df[df['day_name'] == sel_day]
            num_specific_days = day_data['date'].nunique()
            hourly_orders = day_data.groupby('hour')['order_id'].nunique() / num_specific_days
            
            # Tạo bảng lịch trình
            schedule = []
            for hr in range(10, 24):
                avg_o = hourly_orders.get(hr, 0)
                # Công thức: 1 người cân 6 đơn, +1 người dự phòng
                staff_needed = int(np.ceil(avg_o / 6)) + 1 if avg_o > 0 else 0
                
                status = "🔴 Cao điểm" if avg_o > 5 else "🟢 Bình thường" if avg_o > 0 else "⚪ Đóng cửa"
                
                schedule.append({
                    "Khung Giờ": f"{hr}h:00",
                    "Đơn dự kiến": round(avg_o, 1),
                    "Số nhân sự cần": staff_needed,
                    "Trạng thái": status
                })
            
            st.table(pd.DataFrame(schedule))
            st.info("💡 **Gợi ý:** Nhân sự được tính toán dựa trên công thức: `Ceil(Đơn hàng / 6) + 1` để đảm bảo chất lượng dịch vụ giờ cao điểm.")

    st.divider()
    st.caption("Pizza BI Dashboard v7.1 - Dữ liệu thực tế 2015 quy đổi 25.000 VNĐ")