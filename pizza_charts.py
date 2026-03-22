import matplotlib.pyplot as plt
import seaborn as sns
from pizza_processing import get_cleaned_data # Gọi hàm từ file xử lý
import numpy as np

def create_dashboard(df):
    """Hàm chuyên trách vẽ các biểu đồ phân tích với bố cục đã được tối ưu tối đa về không gian"""
    print("\n--- [ĐANG KHỞI TẠO BIỂU ĐỒ PHÂN TÍCH] ---")
    
    # [KHU VỰC CẤU HÌNH TRỌNG TÂM] 
    TITLE_SIZE = 16       
    LABEL_SIZE = 13       
    TICK_SIZE = 11        
    
    # 1. Khoảng cách chữ so với biểu đồ (Padding)
    TITLE_PAD = 25        
    LABEL_PAD_GEN = 15    
    
    # 2. Khoảng cách giữa các biểu đồ với nhau (Spacings)
    ROW_SPACE = 0.6       
    COL_SPACE = 0.3       

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Biểu đồ theo Giờ (Tối ưu nhân sự)
    hourly = df.groupby('hour')['order_details_id'].count().reset_index()
    sns.lineplot(data=hourly, x='hour', y='order_details_id', ax=axes[0,0], marker='o', color='red', linewidth=2)
    axes[0,0].set_title('LƯU LƯỢNG KHÁCH THEO GIỜ', fontweight='bold', color='darkred', pad=TITLE_PAD, fontsize=TITLE_SIZE)
    axes[0,0].set_xlabel('Giờ trong ngày (Hour)', fontsize=LABEL_SIZE, labelpad=LABEL_PAD_GEN)
    axes[0,0].set_ylabel('Số lượng đơn hàng', fontsize=LABEL_SIZE)
    axes[0,0].tick_params(labelsize=TICK_SIZE)
    axes[0,0].set_xticks(range(0, 24))

    # 2. Biểu đồ theo Thứ (Dự báo nguyên liệu)
    sns.barplot(data=df, x='day_name', y='revenue', estimator=sum, ax=axes[0,1], ci=None, palette='viridis')
    axes[0,1].set_title('DOANH THU THEO THỨ TRONG TUẦN', fontweight='bold', color='darkblue', pad=TITLE_PAD, fontsize=TITLE_SIZE)
    axes[0,1].set_xlabel(None) 
    axes[0,1].set_ylabel('Tổng doanh thu ($)', fontsize=LABEL_SIZE)
    axes[0,1].tick_params(axis='x', rotation=30, labelsize=TICK_SIZE)
    axes[0,1].tick_params(axis='y', labelsize=TICK_SIZE)

    # 3. Top 10 Pizza bán chạy (Phân tích Menu)
    top_10 = df.groupby('pizza_name')['quantity'].sum().sort_values(ascending=False).head(10)
    top_10.plot(kind='barh', ax=axes[1,0], color='skyblue')
    axes[1,0].set_title('TOP 10 PIZZA BÁN CHẠY NHẤT', fontweight='bold', pad=TITLE_PAD, fontsize=TITLE_SIZE)
    axes[1,0].set_xlabel('Tổng số lượng bán ra', fontsize=LABEL_SIZE, labelpad=LABEL_PAD_GEN)
    axes[1,0].set_ylabel('Tên Pizza', fontsize=LABEL_SIZE)
    axes[1,0].tick_params(labelsize=TICK_SIZE)
    axes[1,0].invert_yaxis()

    # 4. Doanh thu theo Loại (Market Share)
    category_revenue = df.groupby('category')['revenue'].sum()
    axes[1,1].set_aspect('equal') 
    category_revenue.plot(kind='pie', autopct='%1.1f%%', ax=axes[1,1], cmap='Pastel1', 
                          startangle=140, explode=[0.05]*len(category_revenue), 
                          textprops={'fontsize': 12})
    
    axes[1,1].set_title('TỶ LỆ DOANH THU THEO DANH MỤC', fontweight='bold', pad=TITLE_PAD - 10, fontsize=TITLE_SIZE)
    axes[1,1].set_ylabel('')

    plt.tight_layout(pad=7.0) 
    plt.subplots_adjust(hspace=ROW_SPACE, wspace=COL_SPACE) 
    
    print("📈 Đang hiển thị biểu đồ...")
    plt.show()

def print_business_summary(df):
    """Hàm tính toán các chỉ số kinh doanh cốt lõi để trình bày kết luận"""
    print("\n" + "="*60)
    print("📋 BÁO CÁO TÓM TẮT CHIẾN LƯỢC (DÀNH CHO THUYẾT TRÌNH)")
    print("="*60)

    # 1. Kết luận Doanh thu
    total_rev = df['revenue'].sum()
    peak_day = df.groupby('day_name')['revenue'].sum().idxmax()
    print(f"💰 [DOANH THU]: Tổng đạt ${total_rev:,.2f}.")
    print(f"   👉 Kết luận: Ngày {peak_day} mang lại doanh thu cao nhất. Cần đẩy mạnh Marketing vào các ngày này.")

    # 2. Kết luận Nhân sự
    num_days = df['date'].nunique()
    peak_hour = df.groupby('hour')['order_id'].nunique().idxmax()
    avg_orders_peak = df.groupby('hour')['order_id'].nunique().max() / num_days
    needed_staff = int(np.ceil(avg_orders_peak / 6)) + 1
    print(f"👥 [NHÂN SỰ]: Giờ cao điểm lúc {peak_hour}:00 với trung bình {avg_orders_peak:.1f} đơn/giờ.")
    print(f"   👉 Kết luận: Khuyến nghị bố trí ít nhất {needed_staff} nhân viên vào khung giờ này để đảm bảo vận hành.")

    # 3. Kết luận Kho bãi (Menu)
    best_seller = df.groupby('pizza_name')['quantity'].sum().idxmax()
    best_cat = df.groupby('category')['revenue'].sum().idxmax()
    print(f"📦 [KHO BÃI]: '{best_seller}' là món bán chạy nhất. Loại '{best_cat}' đóng góp doanh thu lớn nhất.")
    print(f"   👉 Kết luận: Cần ưu tiên diện tích kho và tốc độ nhập hàng cho nhóm '{best_cat}'.")
    print("="*60)

if __name__ == "__main__":
    clean_df = get_cleaned_data()
    if clean_df is not None:
        # Vẽ biểu đồ để giảng viên xem trực quan
        create_dashboard(clean_df)
        
        # In ra các con số kết luận để bạn đọc lúc thuyết trình
        print_business_summary(clean_df)