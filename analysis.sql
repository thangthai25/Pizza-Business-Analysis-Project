--luu luong khach theo gio--
SELECT 
    strftime('%H', time) AS order_hour, -- Trích xuất giờ từ cột time
    COUNT(order_id) AS total_orders
FROM orders[cite: 1]
GROUP BY order_hour
ORDER BY order_hour;
--ti le doanh muc 
SELECT 
    pt.category,
    COUNT(DISTINCT od.order_id) AS total_orders,
    SUM(od.quantity * p.price) AS revenue_by_category,
    ROUND(SUM(od.quantity * p.price) * 100.0 / 
        (SELECT SUM(od2.quantity * p2.price) 
         FROM order_details od2 
         JOIN pizzas p2 ON od2.pizza_id = p2.pizza_id), 2) AS revenue_percentage
FROM order_details od
JOIN pizzas p ON od.pizza_id = p.pizza_id[cite: 1]
JOIN pizza_types pt ON p.pizza_type_id = pt.pizza_type_id[cite: 1]
GROUP BY pt.category
ORDER BY revenue_by_category DESC;

--size banh khach an 

SELECT 
    p.size,
    COUNT(od.order_id) AS order_count,
    SUM(od.quantity) AS total_quantity,
    CASE 
        WHEN p.size = 'S' THEN 'Small'
        WHEN p.size = 'M' THEN 'Medium'
        WHEN p.size = 'L' THEN 'Large'
        WHEN p.size = 'XL' THEN 'Extra Large'
        ELSE p.size 
    END AS size_label[cite: 1]
FROM order_details od
JOIN pizzas p ON od.pizza_id = p.pizza_id[cite: 1]
GROUP BY p.size
ORDER BY total_quantity DESC;

--hieu suat van hanh --
SELECT 
    s.name, 
    COUNT(o.order_id) AS orders_handled
FROM staff s
JOIN orders o ON s.staff_id = o.staff_id
GROUP BY s.staff_id
ORDER BY orders_handled DESC;

--nguyen lieu bi lang phi 
SELECT 
    i.ingredient_name, 
    SUM(w.quantity) AS total_wasted_quantity,
    w.reason
FROM waste_logs w
JOIN ingredients i ON w.ingredient_id = i.ingredient_id
GROUP BY i.ingredient_name, w.reason
ORDER BY total_wasted_quantity DESC;

--loi nhuan rong sau khi tru chi phi co dinh
WITH total_sales AS (
    SELECT SUM(od.quantity * p.price) AS revenue FROM order_details od JOIN pizzas p ON od.pizza_id = p.pizza_id
),
total_fixed_costs AS (
    SELECT SUM(amount) AS fixed_expense FROM fixed_costs
)
SELECT 
    revenue, 
    fixed_expense, 
    (revenue - fixed_expense) AS net_profit
FROM total_sales, total_fixed_costs;
