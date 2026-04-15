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

# --- 2. XỬ LÝ DỮ LIỆU SẠCH (DATA PREPROCESSING) ---
@st.cache_data
def load_and_preprocess():
    data_dict = get_cleaned_data()
    if not data_dict: return None
    df = data_dict['main_df']
    
    # Kỹ thuật lọc: Chỉ lấy giờ mở cửa (10h-23h) để tránh nhiễu số 0
    df = df[(df['hour'] >= 10) & (df['hour'] <= 23)].copy()
    
    # Kỹ thuật gộp: Tạo Block 2 giờ để giảm biến động nhỏ
    df['hour_block'] = (df['hour'] // 2) * 2
    
    # Label Encoding cho Thứ trong tuần
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['day_num'] = df['day_name'].map(day_map)
    return df

df = load_and_preprocess()

if df is not None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🍕 PIZZA AI SOLUTIONS")
        module = st.radio("CHỌN MODULE:", ["📊 Dự báo Hiệu suất", "🛒 Robot Gợi ý Upsell"])
        st.divider()
        st.success("Kết nối SQL: Hoàn tất")
        st.caption("Thai Quang Thang - 2026")

    # --- MODULE 1: DỰ BÁO HIỆU SUẤT (FIX CA GÃY & GIỜ LẺ) ---
    if module == "📊 Dự báo Hiệu suất":
        st.title("📊 Hệ thống Điều phối Nhân sự & Hiệu suất")
        
       # --- MODULE 1: DỰ BÁO HIỆU SUẤT & TÀI CHÍNH (FIXED STAFF) ---
    if module == "📊 Dự báo Hiệu suất":
        st.title("📊 Hệ thống Dự báo Doanh thu & Chi phí Nhân sự")
        st.markdown("💡 **Mục tiêu:** Tính toán sự cân bằng giữa Doanh thu dự báo và Chi phí cố định của 6 nhân sự.")

        # 1. Huấn luyện mô hình (Dự báo Số lượng bánh - Quantity)
        ml_data = df.groupby(['date', 'hour_block', 'day_num']).agg({'quantity': 'sum'}).reset_index()
        X_train = ml_data[['hour_block', 'day_num']]
        y_train = ml_data['quantity']

        @st.cache_resource
        def train_model(_X, _y):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(_X.values, _y.values)
            return model

        rf_model = train_model(X_train, y_train)

        # 2. GIAO DIỆN ĐIỀU KHIỂN
        c1, c2 = st.columns(2)
        with c1:
            t_h = st.slider("Chọn giờ kiểm tra:", 10, 22, 11, step=1)
        with c2:
            t_d_name = st.selectbox("Thứ trong tuần:", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            t_d_val = day_map[t_d_name]

        # 3. AI PREDICT & FINANCE LOGIC
        current_block = (t_h // 2) * 2
        input_data = pd.DataFrame([[current_block, t_d_val]], columns=['hour_block', 'day_num'])
        avg_qty = rf_model.predict(input_data.values)[0]
        
        # --- THÔNG SỐ TÀI CHÍNH (GIẢ ĐỊNH THỰC TẾ) ---
        avg_pizza_price = 16.5  # Giá trung bình 1 cái bánh
        hourly_wage = 15.0      # Lương trung bình 1 nhân viên/giờ ($)
        
        # Dự báo doanh thu trong block 2 giờ
        predicted_revenue = avg_qty * avg_pizza_price
        
        # Tính chi phí nhân sự cố định cho 6 người trong 2 giờ
        # Nếu Manager nghỉ ca gãy (14h-17h30), chỉ tính lương cho 5 người
        is_manager_off = (14 <= t_h <= 17)
        active_staff_count = 5 if is_manager_off else 6
        labor_cost = active_staff_count * hourly_wage * 2  # Nhân 2 vì dự báo theo block 2h
        
        # Tỉ lệ chi phí nhân công trên doanh thu (Labor Cost %)
        # Trong F&B, tỉ lệ này đẹp nhất là từ 20% - 30%
        labor_ratio = (labor_cost / predicted_revenue * 100) if predicted_revenue > 0 else 0

        # 4. HIỂN THỊ METRICS TÀI CHÍNH
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Dự báo Doanh thu", f"${predicted_revenue:,.2f}", help="Tính dựa trên số bánh AI dự báo")
        m2.metric("Chi phí Nhân sự (2h)", f"${labor_cost:,.2f}", delta=f"{active_staff_count} người trực")
        m3.metric("Tỉ lệ Cost/Revenue", f"{labor_ratio:.1f}%", delta_color="inverse", delta="Mục tiêu < 30%")

        # 5. BẢNG PHÂN CA & TRẠNG THÁI MANAGER
        col_l, col_r = st.columns([1.5, 1])
        with col_l:
            st.subheader(f"📋 Trạng thái ca trực ({t_d_name} - {t_h}:00)")
            schedule_data = [
                {"Vị trí": "Manager", "Nhân sự": "Quản lý", "Trạng thái": "💤 Nghỉ ca gãy" if is_manager_off else "✅ Đang làm việc"},
                {"Vị trí": "Chef 1", "Nhân sự": "Bếp chính", "Trạng thái": "✅ Đang làm việc"},
                {"Vị trí": "Chef 2", "Nhân sự": "Bếp chính", "Trạng thái": "✅ Đang làm việc"},
                {"Vị trí": "Kitchen Staff", "Nhân sự": "Phụ bếp", "Trạng thái": "✅ Đang làm việc"},
                {"Vị trí": "Waiter 1", "Nhân sự": "Phục vụ", "Trạng thái": "✅ Đang làm việc"},
                {"Vị trí": "Waiter 2", "Nhân sự": "Phục vụ", "Trạng thái": "✅ Đang làm việc"}
            ]
            st.table(pd.DataFrame(schedule_data))

        with col_r:
            st.subheader("📈 Đánh giá Hiệu suất Tài chính")
            if labor_ratio > 40:
                st.error(f"Lương chiếm {labor_ratio:.1f}% doanh thu: Hiệu suất thấp! Quán đang bù lỗ tiền lương.")
            elif 20 <= labor_ratio <= 35:
                st.success(f"Tỉ lệ {labor_ratio:.1f}%: Hiệu suất tối ưu. Lợi nhuận gộp tốt.")
            else:
                st.warning(f"Tỉ lệ {labor_ratio:.1f}%: Cảnh báo thiếu nhân sự cho lượng khách này!")

            # Biểu đồ so sánh
            finance_df = pd.DataFrame({
                'Loại': ['Doanh thu', 'Chi phí Lương'],
                'Số tiền ($)': [predicted_revenue, labor_cost]
            })
            st.plotly_chart(px.bar(finance_df, x='Loại', y='Số tiền ($)', color='Loại', color_discrete_map={'Doanh thu':'#22c55e', 'Chi phí Lương':'#ef4444'}), use_container_width=True)

    # --- MODULE 2: APRIORI (Giao diện chuẩn Admin) ---
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
                else: st.warning("Không tìm thấy quy luật thỏa mãn.")
            else: st.warning("Vui lòng giảm Support.")

    st.divider()
    st.caption("Hệ thống hỗ trợ quyết định vận hành - Thai Quang Thang - ĐH Phương Đông")
else:
    st.error("Lỗi dữ liệu. Kiểm tra lại kết nối SQL!")