import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from pizza_processing import get_cleaned_data

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="PIZZA AI ENGINE", layout="wide", page_icon="🍕")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    [data-testid="stMetric"] { 
        background: white; border-radius: 12px; padding: 15px !important; 
        border-left: 5px solid #ef4444; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KHAI BÁO BIẾN TOÀN CỤC (FIX LỖI DAY_MAP) ---
DAY_MAP = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

# --- 3. XỬ LÝ DỮ LIỆU SẠCH ---
@st.cache_data
def load_all_data():
    data_dict = get_cleaned_data()
    if not data_dict: 
        return None
    
    df = data_dict['main_df']
    # Chỉ lấy khung giờ hoạt động thực tế (10h-23h)
    df = df[(df['hour'] >= 10) & (df['hour'] <= 23)].copy()
    df['hour_block'] = (df['hour'] // 2) * 2
    df['day_num'] = df['day_name'].map(DAY_MAP)
    
    data_dict['main_df'] = df
    return data_dict

data_dict = load_all_data()

if data_dict is not None:
    df = data_dict['main_df']
    staff_df = data_dict.get('staff')

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🍕 PIZZA AI SOLUTIONS")
        module = st.radio("CHỌN MODULE:", ["📊 Dự báo Hiệu suất", "🛒 Robot Gợi ý Upsell"])
        st.divider()
        st.success("Kết nối SQL: Hoàn tất")
        st.caption("Thai Quang Thang - 2026")

   # --- MODULE 1: DỰ BÁO VẬN HÀNH (HỆ 1 GIỜ - CHUẨN ĐỒ ÁN) ---
    if module == "📊 Dự báo Hiệu suất":
        st.title("📊 Quản trị Dự báo & Chuẩn bị theo Giờ")
        st.markdown("💡 **Cơ chế:** AI dự báo sản lượng theo từng giờ đơn lẻ. Mức đỉnh được lấy từ doanh thu cao nhất lịch sử của giờ đó.")

        # 1. TÍNH GIÁ TRỊ TRUNG BÌNH THỰC TẾ (USD)
        total_rev_sql = (df['quantity'] * df['price']).sum() 
        total_qty_sql = df['quantity'].sum()
        real_aov = total_rev_sql / total_qty_sql 

        # 2. HUẤN LUYỆN RF (DỰ BÁO THEO GIỜ - HOURLY)
        # Nhóm dữ liệu theo Ngày, Giờ và Thứ
        ml_data = df.groupby(['date', 'hour', 'day_num']).agg({'quantity': 'sum'}).reset_index()
        X_train = ml_data[['hour', 'day_num']]
        y_train = ml_data['quantity']

        @st.cache_resource
        def train_rf_model(_X, _y):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(_X.values, _y.values)
            return model
        rf_model = train_rf_model(X_train, y_train)

        # 3. ĐIỀU KHIỂN (Chọn chính xác giờ cần kiểm tra)
        c1, c2 = st.columns(2)
        with c1:
            t_h = st.slider("Chọn giờ kiểm tra:", 10, 22, 12, step=1)
        with c2:
            t_d_name = st.selectbox("Thứ trong tuần:", list(DAY_MAP.keys()))
            t_d_val = DAY_MAP[t_d_name]

        # 4. TÍNH TOÁN 2 CHỈ SỐ ĐỘC LẬP THEO GIỜ
        current_hour = t_h
        input_data = pd.DataFrame([[current_hour, t_d_val]], columns=['hour', 'day_num'])
        
        # A. Doanh thu Trung bình (Dự báo AI cho 1 giờ)
        avg_qty_rf = rf_model.predict(input_data.values)[0]
        revenue_rf_avg = avg_qty_rf * real_aov
        
        # B. Đỉnh doanh thu lịch sử (Lấy Max thực tế của chính GIỜ đó trong quá khứ)
        hist_data = df[(df['hour'] == current_hour) & (df['day_num'] == t_d_val)]
        if not hist_data.empty:
            # Tìm doanh thu cao nhất từng đạt được trong đúng giờ này
            revenue_peak = hist_data.groupby('date').apply(lambda x: (x['quantity'] * x['price']).sum()).max()
        else:
            revenue_peak = revenue_rf_avg * 1.5

        # 5. HIỂN THỊ ĐỐI CHIẾU DOANH THU ($)
        st.divider()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(f"💰 Doanh thu TB giờ {t_h}h", f"${revenue_rf_avg:,.2f}")
        with col_m2:
            st.metric(f"🚀 Đỉnh lịch sử giờ {t_h}h", f"${revenue_peak:,.2f}", 
                      delta=f"{(revenue_peak/revenue_rf_avg if revenue_rf_avg > 0 else 0):.1f}x so với TB")

        # 6. DANH MỤC NGUYÊN LIỆU (PREP-LIST HOURLY)
        st.divider()
        st.subheader(f"👨‍🍳 Danh mục chuẩn bị cho khung {t_h}h - {t_h+1}h")
        
        ingredients_df = data_dict.get('ingredients')
        if ingredients_df is not None:
            qty_peak_est = revenue_peak / real_aov
            prep_data = []
            for _, row in ingredients_df.iterrows():
                ing_name = row.get('ingredient_name', 'Nguyên liệu')
                unit = row.get('unit', 'phần')
                
                prep_data.append({
                    "Nguyên liệu (SQL)": ing_name,
                    "Đơn vị": unit,
                    "Cần làm (Mức AI)": f"{avg_qty_rf * 1.1:.1f}",
                    "Chuẩn bị (Mức Đỉnh)": f"{qty_peak_est * 1.1:.1f}",
                    "Ghi chú": "Sơ chế theo giờ"
                })
            st.table(pd.DataFrame(prep_data))

       
        # 7. NHÂN SỰ (LẤY LƯƠNG $ TỪ SQL)
        st.divider()
        st.subheader(f"📋 Bảng điều phối nhân sự ca {t_h}h")
        if staff_df is not None:
            staff_df['role_clean'] = staff_df['role'].str.strip().str.lower()
            wage_map = staff_df.set_index('role_clean')['hourly_rate'].to_dict()
            
            roster = ['Manager', 'Chef', 'Chef', 'Kitchen Staff', 'Server', 'Server']
            is_manager_off = (14 <= t_h <= 17)
            
            display_staff = []
            for r in roster:
                wage = wage_map.get(r.lower(), 0)
                status = " Trong ca"
                if r == 'Manager' and is_manager_off:
                    status = " Nghỉ ca gãy"; wage = 0
                display_staff.append({"Vị trí": r, "Lương ($)": f"${wage:.2f}", "Trạng thái": status})
            st.table(pd.DataFrame(display_staff))

    # --- MODULE 2: APRIORI (Gợi ý Upsell) ---
    elif module == "🛒 Robot Gợi ý Upsell":
        st.title("🛒 Khai phá Quy luật Mua sắm (MBA)")
        col_res, col_ctrl = st.columns([2.5, 1], gap="large")

        with col_ctrl:
            st.subheader("⚙️ Cấu hình AI")
            m_sup = st.slider("Support (Độ phổ biến)", 0.001, 0.05, 0.01, format="%.3f")
            m_conf = st.slider("Confidence (Độ tin cậy)", 0.1, 1.0, 0.3)
            m_lift = st.number_input("Lift (Sức mạnh liên kết)", 1.0, 5.0, 1.2)

        with col_res:
            basket = df.groupby(['order_id', 'pizza_name'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('order_id')
            basket_sets = basket.map(lambda x: 1 if x >= 1 else 0)
            frequent = apriori(basket_sets, min_support=m_sup, use_colnames=True)
            if not frequent.empty:
                rules = association_rules(frequent, metric="confidence", min_threshold=m_conf)
                rules = rules[rules['lift'] >= m_lift]
                if not rules.empty:
                    rules['Gốc'] = rules['antecedents'].apply(lambda x: list(x)[0])
                    rules['Gợi ý'] = rules['consequents'].apply(lambda x: list(x)[0])
                    st.success(f"Tìm thấy {len(rules)} quy luật!")
                    st.dataframe(rules[['Gốc', 'Gợi ý', 'support', 'confidence', 'lift']], use_container_width=True)
                else: 
                    st.warning("Không tìm thấy quy luật thỏa mãn.")
            else: 
                st.warning("Vui lòng giảm Support.")

    st.divider()
    st.caption("Hệ thống hỗ trợ quyết định vận hành - Thai Quang Thang - ĐH Phương Đông")
else:
    st.error("Lỗi dữ liệu. Kiểm tra lại kết nối SQL!")