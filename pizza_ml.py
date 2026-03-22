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
    
    # Kiểm tra xem cột order_id có tồn tại không để tránh lỗi KeyError
    if 'order_id' not in df.columns:
        print("❌ Lỗi: Không tìm thấy cột 'order_id'. Vui lòng chạy lại file SQL View để cập nhật dữ liệu.")
        return pd.DataFrame()

    # Chuẩn bị dữ liệu dạng Basket (Giỏ hàng)
    basket = (df.groupby(['order_id', 'pizza_name'])['quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('order_id'))
    
    # Chuyển đổi sang dạng One-Hot Encoding (0 hoặc 1)
    def encode_units(x):
        return 1 if x >= 1 else 0
    
    # FIX LỖI: Sử dụng .map() thay vì .applymap() cho Pandas phiên bản mới (Python 3.13)
    basket_sets = basket.map(encode_units)
    
    # Chạy thuật toán Apriori với min_support = 1%
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("⚠️ Không tìm thấy tập phổ biến nào với min_support hiện tại.")
        return pd.DataFrame()

    # Tạo luật kết hợp (Association Rules)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # Sắp xếp theo độ tin cậy (Confidence) và sức mạnh liên kết (Lift)
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False)
    
    print(f"✅ Đã tìm thấy {len(rules)} quy luật kết hợp.")
    return rules

def run_time_series_forecasting(df, target='orders'):
    """
    Áp dụng Time Series Forecasting (Dùng Random Forest Regressor)
    Mục tiêu: Dự báo số lượng đơn hàng hoặc Doanh thu theo ngày/giờ.
    """
    label = "ĐƠN HÀNG" if target == 'orders' else "DOANH THU"
    print(f"\n--- [2. TIME SERIES FORECASTING - DỰ BÁO {label}] ---")
    
    # 1. Gom nhóm dữ liệu theo Ngày và Giờ
    if target == 'orders':
        # Dự báo số lượng đơn hàng (Dựa trên unique order_id)
        data_group = df.groupby(['date', 'hour']).agg({'order_id': 'nunique'}).reset_index()
        data_group.columns = ['date', 'hour', 'y']
    else:
        # Dự báo doanh thu (Dựa trên cột revenue)
        data_group = df.groupby(['date', 'hour']).agg({'revenue': 'sum'}).reset_index()
        data_group.columns = ['date', 'hour', 'y']
    
    # 2. Tạo Features (Đặc trưng thời gian)
    data_group['day_of_week'] = data_group['date'].dt.dayofweek
    data_group['month'] = data_group['date'].dt.month
    data_group['is_weekend'] = data_group['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3. Chia dữ liệu Train/Test
    X = data_group[['hour', 'day_of_week', 'month', 'is_weekend']]
    y = data_group['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Huấn luyện mô hình Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Đánh giá mô hình
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Mô hình dự báo {label} hoàn tất.")
    print(f"   - Sai số trung bình (MAE): {mae:.2f}")
    print(f"   - Độ phù hợp (R2 Score): {r2:.2f}")
    
    return model

if __name__ == "__main__":
    # Lấy dữ liệu sạch từ file processing
    data = get_cleaned_data()
    
    if data is not None:
        # --- NHIỆM VỤ 1 & 4: APRIORI & COMBO GỢI Ý ---
        rules = run_market_basket_analysis(data)
        
        if not rules.empty:
            print("\n🔥 GỢI Ý COMBO (Top 5 cặp liên quan nhất):")
            for idx, row in rules.head(5).iterrows():
                items_a = ', '.join(list(row['antecedents']))
                items_b = ', '.join(list(row['consequents']))
                print(f"👉 Nếu khách mua [{items_a}] -> Khả năng cao sẽ mua [{items_b}]")
                print(f"   (Độ tin cậy: {row['confidence']*100:.1f}% | Lift: {row['lift']:.2f})")
            
        #  NHIỆM VỤ 2: DỰ BÁO ĐƠN HÀNG (SỐ LƯỢNG) 
        order_model = run_time_series_forecasting(data, target='orders')
        
        #  NHIỆM VỤ 3: DỰ BÁO DOANH THU (TIỀN) 
        revenue_model = run_time_series_forecasting(data, target='revenue')
        
        #  DEMO DỰ BÁO CHO NGÀY MAI 
        # Giả sử: 19h (Giờ cao điểm tối), Thứ 7 (Day 5), Tháng 12, Weekend=1
        sample_input = pd.DataFrame([[19, 5, 12, 1]], columns=['hour', 'day_of_week', 'month', 'is_weekend'])
        
        pred_orders = order_model.predict(sample_input)[0]
        pred_revenue = revenue_model.predict(sample_input)[0]
        
        print("\n" + "="*50)
        print("🔮 KẾT QUẢ DỰ BÁO KỊCH BẢN (19h Tối Thứ 7 - Tháng 12):")
        print(f"📈 Dự kiến số đơn hàng: {int(pred_orders)} đơn")
        print(f"💰 Dự kiến doanh thu: ${pred_revenue:,.2f}")
        print("="*50)