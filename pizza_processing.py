import pyodbc
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def get_cleaned_data():
    """Hàm tổng hợp: Kết nối -> Lấy dữ liệu -> Làm sạch"""
    
    # 1. Kết nối SQL Server
    conn_str = (
        'Driver={SQL Server};'
        'Server=.;'
        'Database=PizzaProject;'
        'Trusted_Connection=yes;'
    )
    
    try:
        conn = pyodbc.connect(conn_str)
        print("✅ Kết nối SQL Server thành công!")
        
        # 2. Lấy dữ liệu từ View
        raw_df = pd.read_sql("SELECT * FROM Pizza_Master_Data", conn)
        conn.close()
        
        # 3. Làm sạch dữ liệu
        df = raw_df.copy()
        
        # Xử lý giá trị thiếu & trùng lặp
        df = df.dropna().drop_duplicates()
        
        # Chuẩn hóa thời gian
        df['date'] = pd.to_datetime(df['date'])
        
        # --- PHẦN CHỈNH SỬA ĐỊNH DẠNG TIME TẠI ĐÂY ---
        # Chuyển đổi sang datetime tạm thời để xử lý
        time_temp = pd.to_datetime(df['time'])
        
        # 1. Lấy giờ (dạng số) để vẽ biểu đồ và làm ML
        df['hour'] = time_temp.dt.hour
        
        # 2. Định dạng lại cột time thành chuỗi sạch HH:MM:SS (Xóa bỏ .0000000)
        df['time'] = time_temp.dt.strftime('%H:%M:%S')
        # ---------------------------------------------
        
        df['day_name'] = df['date'].dt.day_name()
        
        # Sắp xếp thứ tự thứ trong tuần
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
       # ... (giữ nguyên phần trên) ...
        
        df['day_name'] = pd.Categorical(df['day_name'], categories=days_order, ordered=True)

        # --- TÍNH TOÁN NHÂN SỰ ĐỒNG BỘ TẠI ĐÂY ---
        # Đếm số đơn hàng duy nhất (order_id) theo từng giờ
        hourly_orders = df.groupby('hour')['order_id'].nunique().to_dict()
        
        # Công thức: Cứ 10 đơn hàng/giờ cần 1 nhân viên (tối thiểu 2 người/ca)
        # Việc tính ở đây giúp mốc 12h (từ 12:00:00 - 12:59:59) luôn chính xác
        df['staff_suggested'] = df['hour'].map(lambda x: max(2, hourly_orders.get(x, 0) // 10))
        
        # Quy đổi doanh thu sang VNĐ
        df['revenue'] = df['revenue'] * 25000
        
        print(f"✅ Đã chốt số liệu nhân sự và quy đổi doanh thu.")
        return df
        
    except Exception as e:
        print(f" Lỗi trong quá trình xử lý: {e}")
        return None

if __name__ == "__main__":
    # Chạy thử để kiểm tra
    data = get_cleaned_data()
    if data is not None:
        print(data.head())