import pyodbc
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def get_sql_connection():
    conn_str = (
        'Driver={SQL Server};'
        'Server=.;' 
        'Database=PizzaProject;' 
        'Trusted_Connection=yes;'
    )
    return pyodbc.connect(conn_str)

def get_cleaned_data():
    try:
        conn = get_sql_connection()
        
        # 1. Lấy 4 bảng gốc (Thêm pizza_types)
        orders = pd.read_sql("SELECT * FROM orders", conn)
        order_details = pd.read_sql("SELECT * FROM order_details", conn)
        pizzas = pd.read_sql("SELECT * FROM pizzas", conn)
        pizza_types = pd.read_sql("SELECT * FROM pizza_types", conn) # <-- QUAN TRỌNG
        
        # 2. Lấy 5 bảng vận hành
        staff = pd.read_sql("SELECT * FROM staff", conn)
        shift_log = pd.read_sql("SELECT * FROM shift_logs", conn)
        fixed_costs = pd.read_sql("SELECT * FROM fixed_costs", conn)
        waste_log = pd.read_sql("SELECT * FROM waste_logs", conn)
        ingredients = pd.read_sql("SELECT * FROM ingredients", conn)
        
        conn.close()

        # MERGE DỮ LIỆU: Phải nối thêm bảng pizza_types để lấy tên
        df = order_details.merge(pizzas, on='pizza_id') \
                          .merge(orders, on='order_id') \
                          .merge(pizza_types, on='pizza_type_id') # <-- Lấy tên ở đây

        # Đổi tên cột 'name' từ SQL thành 'pizza_name' để khớp với code pizza_app.py
        if 'name' in df.columns:
            df = df.rename(columns={'name': 'pizza_name'})

        df['date'] = pd.to_datetime(df['date'])
        time_temp = pd.to_datetime(df['time'])
        df['hour'] = time_temp.dt.hour
        df['time'] = time_temp.dt.strftime('%H:%M:%S')
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_name'] = pd.Categorical(df['date'].dt.day_name(), categories=days_order, ordered=True)
        
        df['revenue_vnd'] = df['quantity'] * df['price'] * 25000
        
        return {
            'main_df': df,
            'staff': staff,
            'shift': shift_log,
            'fixed': fixed_costs,
            'waste': waste_log,
            'ingredients': ingredients
        }
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None