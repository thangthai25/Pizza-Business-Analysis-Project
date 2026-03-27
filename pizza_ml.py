import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pizza_processing import get_cleaned_data

def run_market_basket_analysis(df):
    """
    Áp dụng thuật toán Apriori (Market Basket Analysis)
    Mục tiêu: Tìm mối liên quan giữa các món ăn để gợi ý Combo.
    """
    print("\n" + "="*50)
    print("--- [1. MARKET BASKET ANALYSIS (APRIORI)] ---")
    
    if 'order_id' not in df.columns:
        print("❌ Lỗi: Không tìm thấy cột 'order_id'.")
        return pd.DataFrame()

    # Chuẩn bị dữ liệu dạng Basket
    basket = (df.groupby(['order_id', 'pizza_name'])['quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('order_id'))
    
    basket_sets = basket.map(lambda x: 1 if x >= 1 else 0)
    
    # Chạy thuật toán Apriori
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame()

    # Tạo luật kết hợp
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False)
    
    print(f"✅ Tìm thấy {len(rules)} quy luật kết hợp.")
    return rules

def run_time_series_forecasting(df, target='revenue'):
    """
    Áp dụng Time Series Forecasting (Random Forest) với Feature Engineering nâng cao.
    Mục tiêu: Dự báo doanh thu/đơn hàng và phân tích các yếu tố ảnh hưởng.
    """
    label = "DOANH THU (VNĐ)" if target == 'revenue' else "ĐƠN HÀNG"
    print(f"\n--- [2. DỰ BÁO {label} & PHÂN TÍCH ĐẶC TRƯNG] ---")
    
    # 1. Gom nhóm dữ liệu theo Ngày và Giờ
    if target == 'revenue':
        data_group = df.groupby(['date', 'hour']).agg({'revenue': 'sum'}).reset_index()
    else:
        data_group = df.groupby(['date', 'hour']).agg({'order_id': 'nunique'}).reset_index()
    
    data_group.columns = ['date', 'hour', 'y']
    data_group = data_group.sort_values(['date', 'hour'])

    # 2. FEATURE ENGINEERING
    data_group['day_of_week'] = data_group['date'].dt.dayofweek
    data_group['month'] = data_group['date'].dt.month
    data_group['is_weekend'] = data_group['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Thêm biến trễ (Lag Features): Doanh thu/Đơn hàng của 1 giờ trước đó
    data_group['lag_1h'] = data_group['y'].shift(1).fillna(data_group['y'].median())
    
    # 3. Chia dữ liệu
    features = ['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1h']
    X = data_group[features]
    y = data_group['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Huấn luyện mô hình Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Đánh giá mô hình
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 6. Tính toán Tầm quan trọng của đặc trưng
    importance_df = pd.DataFrame({
        'Yếu tố': features,
        'Mức độ ảnh hưởng (%)': model.feature_importances_ * 100
    }).sort_values(by='Mức độ ảnh hưởng (%)', ascending=False)
    
    print(f"✅ Mô hình {label} hoàn tất. Độ phù hợp (R2): {r2:.2f}")
    if target == 'revenue':
        print(f"   - Sai số trung bình (MAE): {mae:,.0f} VNĐ")
    
    return model, importance_df

if __name__ == "__main__":
    # Lấy dữ liệu sạch (Đã nhân tỷ giá 25.000 từ file processing)
    data = get_cleaned_data()
    
    if data is not None:
        # NHIỆM VỤ 1: LUẬT KẾT HỢP (COMBO)
        rules = run_market_basket_analysis(data)
        if not rules.empty:
            print("\n🔥 TOP 3 COMBO GỢI Ý MẠNH NHẤT:")
            for _, row in rules.head(3).iterrows():
                print(f"👉 Mua [{', '.join(list(row['antecedents']))}] -> Thường mua [{', '.join(list(row['consequents']))}]")

        # NHIỆM VỤ 2: DỰ BÁO DOANH THU & PHÂN TÍCH ẢNH HƯỞNG
        rev_model, importance = run_time_series_forecasting(data, target='revenue')
        
        print("\n📊 CÁC YẾU TỐ QUYẾT ĐỊNH DOANH THU:")
        print(importance.to_string(index=False))

        # DEMO DỰ BÁO KỊCH BẢN
        # 19h tối Thứ 7, tháng 12, Weekend=1, 
        # Giả sử doanh thu 1h trước (lag_1h) là 5.000.000 VNĐ (đã đổi sang VNĐ)
        lag_val_demo = 5000000 
        sample_input = pd.DataFrame([[19, 5, 12, 1, lag_val_demo]], 
                                    columns=['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1h'])
        
        prediction = rev_model.predict(sample_input)[0]
        
        print("\n" + "="*60)
        print(f"🔮 DỰ BÁO KỊCH BẢN (19h Tối Thứ 7 - Tháng 12):")
        print(f"💰 Doanh thu dự kiến: {prediction:,.0f} VNĐ")
        print("="*60)