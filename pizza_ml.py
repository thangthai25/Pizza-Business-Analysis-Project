import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from pizza_processing import get_cleaned_data

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="PIZZA AI BRAIN | Ver 8.0", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    [data-testid="stMetric"] {
        background: white; border-radius: 12px; padding: 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TẢI DỮ LIỆU TỪ SQL ---
@st.cache_resource
def load_all_data():
    data = get_cleaned_data()
    if data:
        df = data['main_df']
        # Chuẩn bị tập ML cho dự báo doanh thu theo giờ
        df_ml = df.groupby(['date', 'hour']).agg({'revenue_vnd': 'sum'}).reset_index()
        df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
        df_ml['month'] = df_ml['date'].dt.month
        df_ml['is_weekend'] = df_ml['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_ml['lag_1h'] = df_ml['revenue_vnd'].shift(1).fillna(df_ml['revenue_vnd'].median())
        return data, df_ml
    return None, None

data_dict, df_ml = load_all_data()

if data_dict:
    df = data_dict['main_df']
    staff = data_dict['staff']
    shift = data_dict['shift']
    waste = data_dict['waste']
    ingredients = data_dict['ingredients']

    # --- SIDEBAR ĐIỀU KHIỂN ---
    with st.sidebar:
        st.title("🍕 PIZZA AI BRAIN")
        st.subheader("Hệ thống học máy SQL-Ready")
        mode = st.radio("CHỌN MÔ HÌNH HỌC MÁY:", [
            "🔮 Mô hình 1: Dự báo Doanh thu (Random Forest)",
            "🎯 Mô hình 2: Phân nhóm Sản phẩm (K-Means)",
            "🛒 Mô hình 3: Gợi ý Combo (Association Rules)",
            "👥 Mô hình 4: Tối ưu Nhân sự (Staff Analytics)"
        ])
        st.divider()
        st.caption("Data Source: SQL Server 2015 | Tỷ giá: 25.000đ")

    # --- MÔ HÌNH 1: DỰ BÁO DOANH THU ---
    if mode == "🔮 Mô hình 1: Dự báo Doanh thu (Random Forest)":
        st.title("🔮 AI Dự báo Doanh thu & Kịch bản Giả định")
        
        # Huấn luyện mô hình
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1h']
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df_ml[features], df_ml['revenue_vnd'])
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("🧪 Chạy kịch bản What-if")
            in_h = st.slider("Giờ làm việc", 10, 23, 19)
            in_d = st.selectbox("Thứ", range(7), format_func=lambda x: ['T2','T3','T4','T5','T6','T7','CN'][x])
            in_m = st.slider("Tháng", 1, 12, 12)
            
            if st.button("🚀 Chạy AI Forecast"):
                pred = model.predict([[in_h, in_d, in_m, 1 if in_d >= 5 else 0, df_ml['revenue_vnd'].mean()]])[0]
                st.metric("Dự báo doanh thu", f"{pred:,.0f} VNĐ")
                st.write(f"Định mức chuẩn bị: ~{int(pred/250000)} cái bánh.")

        with c2:
            st.subheader("📈 Tầm quan trọng của các yếu tố")
            feat_imp = pd.DataFrame({'Yếu tố': ['Giờ', 'Thứ', 'Tháng', 'Cuối tuần', 'Doanh thu trước'], 'Độ ảnh hưởng': model.feature_importances_})
            st.plotly_chart(px.bar(feat_imp, x='Độ ảnh hưởng', y='Yếu tố', orientation='h', color='Độ ảnh hưởng'))

    # --- MÔ HÌNH 2: PHÂN NHÓM SẢN PHẨM ---
    elif mode == "🎯 Mô hình 2: Phân nhóm Sản phẩm (K-Means)":
        st.title("🎯 Chiến lược Thực đơn (Ma trận BCG - K-Means)")
        p_data = df.groupby('pizza_name').agg({'revenue_vnd': 'sum', 'quantity': 'sum', 'price': 'mean'}).reset_index()
        
        # Chuẩn hóa dữ liệu để K-Means chạy chuẩn
        scaler = StandardScaler()
        p_scaled = scaler.fit_transform(p_data[['quantity', 'revenue_vnd']])
        
        p_data['cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(p_scaled)
        
        fig = px.scatter(p_data, x="quantity", y="revenue_vnd", color="cluster", 
                         size="revenue_vnd", hover_name="pizza_name",
                         title="Phân nhóm 4 loại Pizza dựa trên Doanh thu & Sản lượng")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Nhóm 0 (Ngôi sao):** Doanh thu cao, sản lượng lớn. **Nhóm 3 (Bò sữa):** Giá cao, doanh thu tốt nhưng kén người mua.")

    # --- MÔ HÌNH 3: GỢI Ý COMBO ---
    elif mode == "🛒 Mô hình 3: Gợi ý Combo (Association Rules)":
        st.title("🛒 Khám phá hành vi mua sắm (Market Basket Analysis)")
        
        # Xử lý ma trận giỏ hàng
        basket = df.groupby(['order_id', 'pizza_name'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('order_id')
        basket_sets = basket.map(lambda x: 1 if x >= 1 else 0)
        
        frequent = apriori(basket_sets, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent, metric="lift", min_threshold=1.2)
        
        if not rules.empty:
            rules['Món A'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['Món B'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            st.dataframe(rules[['Món A', 'Món B', 'confidence', 'lift']].sort_values('lift', ascending=False), use_container_width=True)
        else:
            st.warning("Không tìm thấy combo phổ biến nào với Support này.")

    # --- MÔ HÌNH 4: TỐI ƯU NHÂN SỰ ---
    elif mode == "👥 Mô hình 4: Tối ưu Nhân sự (Staff Analytics)":
        st.title("👥 Quản trị Nhân lực & Hiệu suất")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Phân bổ giờ làm thực tế từ SQL")
            perf = shift.merge(staff, on='staff_id').groupby('name')['hour_worked'].sum().reset_index()
            st.plotly_chart(px.pie(perf, values='hour_worked', names='name', hole=.3))
            
        with c2:
            st.subheader("Chi phí lương theo vai trò")
            role_pay = shift.merge(staff, on='staff_id')
            role_pay['total_pay'] = role_pay['hour_worked'] * role_pay['hourly_rate']
            role_summary = role_pay.groupby('_role')['total_pay'].sum().reset_index()
            st.plotly_chart(px.bar(role_summary, x='_role', y='total_pay', color='_role'))

    st.divider()
    st.caption("AI Engine v8.0 - Tối ưu hóa trên nền tảng SQL Server")
else:
    st.error("Lỗi kết nối SQL. Vui lòng kiểm tra lại file pizza_processing.py")