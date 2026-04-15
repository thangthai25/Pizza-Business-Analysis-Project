import pandas as pd
import pyodbc
import os

# --- 1. CẤU HÌNH KẾT NỐI (NHÌN KỸ DẤU HAI CHẤM : SAU MỖI CHỮ) ---
DB_CONFIG = {
    "driver": "{SQL Server}",
    "server": ".",
    "database": "PizzaProject",
    "trusted_connection": "yes"
}

def get_sql_connection():
    # Lấy thông tin từ Dictionary bằng Key
    driver = DB_CONFIG["driver"]
    server = DB_CONFIG["server"]
    database = DB_CONFIG["database"]
    trusted = DB_CONFIG["trusted_connection"]
    
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted};"
    return pyodbc.connect(conn_str)

def export_to_csv():
    # Tạo thư mục chứa dữ liệu
    export_dir = "tableau_data_source"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"✔️ Đã tạo thư mục: {export_dir}")

    # Danh sách các bảng cần xuất (Phải dùng dấu : ở giữa)
    queries = {
        "staff": "SELECT * FROM staff",
        "shift_logs": "SELECT * FROM shift_logs",
        "fixed_costs": "SELECT * FROM fixed_costs",
        "waste_logs": "SELECT * FROM waste_logs",
        "ingredients": "SELECT * FROM ingredients",
        "recipes":"SELECT*FROM recipes",
        "supplier" :"SELECT*FROM suppliers",
    }

    try:
        conn = get_sql_connection()
        print("🔗 Kết nối SQL Server thành công!")

        for file_name, sql in queries.items():
            print(f"⏳ Đang trích xuất bảng: {file_name}...")
            df = pd.read_sql(sql, conn)
            
            # Xuất file CSV (utf-8-sig để không lỗi font tiếng Việt trong Tableau)
            file_path = os.path.join(export_dir, f"{file_name}.csv")
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✅ Đã lưu: {file_path}")

        print("\n🚀 HOÀN TẤT! Dữ liệu đã sẵn sàng cho Tableau.")
        conn.close()

    except Exception as e:
        print(f"❌ Lỗi chi tiết: {str(e)}")

if __name__ == "__main__":
    export_to_csv()