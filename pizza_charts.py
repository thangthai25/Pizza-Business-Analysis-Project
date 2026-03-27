import matplotlib.pyplot as plt
import seaborn as sns
from pizza_processing import get_cleaned_data
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

# --- HÀM HỖ TRỢ ĐỊNH DẠNG SỐ TIỀN VNĐ (TRIỆU/TỶ) ---
def format_func(x, pos):
    if x >= 1_000_000_000:
        return f'{x*1e-9:.1f} Tỷ'
    elif x >= 1_000_000:
        return f'{x*1e-6:.1f} Tr'
    return f'{x:,.0f}'

def plot_revenue_timeline(df):
    """
    Biểu đồ cột Doanh thu theo Tháng (VNĐ).
    Chứng minh mốc bắt đầu từ Tháng 1/2015.
    """
    print("\n--- [ĐANG TẠO BIỂU ĐỒ DOANH THU THEO THÁNG - ĐƠN VỊ VNĐ] ---")
    
    # 1. Chuẩn bị dữ liệu
    df_monthly = df.copy()
    
    # Tạo cột Tháng (Dạng số 1, 2, 3...)
    df_monthly['month_num'] = df_monthly['date'].dt.month
    
    # Gom nhóm tính tổng doanh thu mỗi tháng
    monthly_data = df_monthly.groupby('month_num')['revenue'].sum().reset_index()
    
    # Tạo nhãn hiển thị "Tháng 1", "Tháng 2"...
    monthly_data['month_label'] = monthly_data['month_num'].apply(lambda x: f"Th. {int(x)}")

    # 2. Vẽ biểu đồ Cột (Bar Chart)
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.set_style("white")

    # Vẽ cột với hiệu ứng màu Reds chuyên nghiệp
    colors = sns.color_palette("Reds", n_colors=12)
    bars = sns.barplot(data=monthly_data, x='month_label', y='revenue', palette=colors, ax=ax)

    # 3. THÊM SỐ LIỆU TRÊN ĐẦU MỖI CỘT (VNĐ rút gọn)
    for p in bars.patches:
        ax.annotate(format_func(p.get_height(), None), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9, fontweight='bold')

    # 4. CHỨNG MINH MỐC THÁNG 1
    # Chỉ mũi tên vào cột đầu tiên (Tháng 1)
    ax.annotate('Khởi đầu: Tháng 1/2015', 
                xy=(0, monthly_data.iloc[0]['revenue']), 
                xytext=(30, 40), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", fc="yellow", alpha=0.3))

    # 5. Định dạng Trục Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # 6. Tiêu đề và Nhãn
    ax.set_title("TỔNG DOANH THU HÀNG THÁNG (DỮ LIỆU GỐC 2015)", fontweight='bold', fontsize=16, pad=30)
    ax.set_xlabel("Trình tự thời gian", fontsize=12)
    ax.set_ylabel("Doanh thu (VNĐ)", fontsize=12)
    
    sns.despine()
    plt.tight_layout()
    plt.show()

def create_dashboard(df):
    """Vẽ Dashboard 4 biểu đồ (Đã bỏ phần nhân tỷ giá dư thừa)"""
    print("\n--- [ĐANG KHỞI TẠO DASHBOARD PHÂN TÍCH] ---")
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Biểu đồ theo Giờ
    hourly = df.groupby('hour')['order_id'].nunique().reset_index()
    sns.lineplot(data=hourly, x='hour', y='order_id', ax=axes[0,0], marker='o', color='red')
    axes[0,0].set_title('LƯU LƯỢNG KHÁCH THEO GIỜ', fontweight='bold', pad=20)

    # 2. Biểu đồ theo Thứ
    sns.barplot(data=df, x='day_name', y='revenue', estimator=sum, ax=axes[0,1], palette='viridis')
    axes[0,1].set_title('DOANH THU THEO THỨ (VNĐ)', fontweight='bold', pad=20)
    axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # 3. Top 10 Pizza
    top_10 = df.groupby('pizza_name')['quantity'].sum().sort_values(ascending=False).head(10)
    top_10.plot(kind='barh', ax=axes[1,0], color='skyblue')
    axes[1,0].set_title('TOP 10 PIZZA BÁN CHẠY', fontweight='bold', pad=0)
    axes[1,0].invert_yaxis()

    # 4. Tỷ lệ doanh thu
    cat_rev = df.groupby('category')['revenue'].sum()
    cat_rev.plot(kind='pie', autopct='%1.1f%%', ax=axes[1,1], cmap='Pastel1', startangle=140)
    axes[1,1].set_title('TỶ LỆ DOANH THU THEO DANH MỤC', fontweight='bold', pad=0-5)
    axes[1,1].set_ylabel('')

    plt.tight_layout(pad=5.0) 
    plt.show()

def print_business_summary(df):
    """In tóm tắt kết luận (Dùng doanh thu đã nhân ở processing)"""
    total_rev = df['revenue'].sum()
    
    print("\n" + "="*60)
    print("📋 BÁO CÁO TÓM TẮT CHIẾN LƯỢC (VNĐ)")
    print("="*60)
    print(f"💰 [DOANH THU]: Tổng đạt {total_rev:,.0f} VNĐ.")
    
    num_days = df['date'].nunique()
    peak_h = df.groupby('hour')['order_id'].nunique().idxmax()
    avg_o = df.groupby('hour')['order_id'].nunique().max() / num_days
    print(f"👥 [NHÂN SỰ]: Cao điểm lúc {peak_h}h. Cần {int(np.ceil(avg_o/6))+1} nhân sự.")
    print("="*60)

if __name__ == "__main__":
    # Lưu ý: File pizza_processing.py của bạn phải có dòng nhân 25000 rồi
    clean_df = get_cleaned_data()
    if clean_df is not None:
        plot_revenue_timeline(clean_df)
        create_dashboard(clean_df)
        print_business_summary(clean_df)