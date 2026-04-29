<img width="1913" height="1038" alt="image" src="https://github.com/user-attachments/assets/43e1f3d4-bb58-4a02-ac2f-da5a0125e16e" /># Dự Án Phân Tích Dữ Liệu Bán Hàng Chuỗi Nhà Hàng 
<img width="1903" height="1032" alt="image" src="https://github.com/user-attachments/assets/598312f1-f2dd-4215-886d-abf47430d26d" />


## 🛠 Công cụ & Ngôn ngữ
**Ngôn ngữ:** Python/SQL
**Thư viện/Framework:** Pandas, Tableau
**Cơ sở dữ liệu:** SQL Server

## 📌 Tổng quan dự án
Thay vì chỉ tập trung vào doanh số, dự án này đi sâu vào bài toán **Lợi nhuận thực tế**. Bằng cách kết hợp dữ liệu bán hàng (Output) với các dữ liệu chi phí vận hành (Input), tôi đã xây dựng một bức tranh toàn diện về sức khỏe tài chính của cửa hàng.

## 2. 📌 Vấn đề & Giải pháp
Dự án không chỉ dừng lại ở việc đếm số bánh bán ra mà tập trung vào **Lợi nhuận thực tế**.
* **Đầu vào (Input):** Dữ liệu bán hàng + Chi phí cố định + Lương nhân sự + Giá nguyên liệu.
<img width="1691" height="833" alt="image" src="https://github.com/user-attachments/assets/48cc7a0d-6f31-405c-b99b-53b8090290c7" />


<img width="1561" height="967" alt="image" src="https://github.com/user-attachments/assets/5c21ddfa-9349-42d5-ad82-609d3d5c96eb" />

<img width="1907" height="721" alt="image" src="https://github.com/user-attachments/assets/1f26fea1-85f3-4a1b-9e30-e950f22ea17c" />



* **Đầu ra (Output):** Dashboard theo dõi sức khỏe tài chính .

<img width="1913" height="1038" alt="image" src="https://github.com/user-attachments/assets/cf6ea415-4d77-4e82-b501-88a0c7d78dab" />

## 3. 🛠️ Quy trình xử lý dữ liệu (Workflow)
1. **data cleaning:** Dùng python làm sạch dữ liệu.
2. **Tích hợp:** kết nối các bảng chi phí tự tạo (Lương, Mặt bằng, Nguyên liệu) vào bộ dữ liệu chính.
<img width="1433" height="718" alt="image" src="https://github.com/user-attachments/assets/96613edf-e068-4217-af90-8da17283cf1c" />

## 4. 📊 Key Insights (Những phát hiện "đắt giá")

###  Doanh thu vs Lợi nhuận

* **Insight:** The Thai Chicken Pizza là 'cỗ máy in tiền' thực sự của cửa hàng khi đứng đầu về doanh thu (xấp xỉ 43k) nhưng lại sử dụng các nguyên liệu có giá vốn rất tối ưu (chỉ từ 2.2 - 4.5); trong khi đó, các dòng Pizza cao cấp sử dụng Prosciutto hay Parmesan Aged dù có doanh số ổn định nhưng lại chịu áp lực cực lớn về biên lợi nhuận do chi phí nguyên liệu đầu vào cao gấp đôi so với nhóm còn lạ.
<img width="1668" height="694" alt="image" src="https://github.com/user-attachments/assets/3c04402d-40f8-4c26-891e-f4b83f55ea7e" />
<img width="1103" height="334" alt="image" src="https://github.com/user-attachments/assets/11d799fa-936c-4e48-9b98-a9d7ba14364a" />

###  Thời điểm vàng
<img width="1654" height="962" alt="image" src="https://github.com/user-attachments/assets/8b01587e-b846-4dd6-b00b-c0c42e2aaeab" />

* **Insight:** Khung giờ từ 12:00 - 13:00 đóng góp 40% doanh thu ngày nhưng cũng là lúc chi phí nhân sự tăng cao nhất.

### Nhân sự 
<img width="1654" height="962" alt="image" src="https://github.com/user-attachments/assets/8b01587e-b846-4dd6-b00b-c0c42e2aaeab" />
Insight: "Dù doanh thu vào khung giờ 14:00 - 16:00 thấp hơn, nhưng tỷ lệ Chi phí nhân sự / Doanh thu lại cao nhất do số lượng nhân viên trực ca vẫn giữ nguyên trong khi đơn hàng ít
Hành động đề xuất: Cân đối lại số lượng nhân viên theo ca (Part-time/Full-time) để giảm lãng phí lương vào giờ thấp điểm.

### Menu
<img width="1284" height="942" alt="image" src="https://github.com/user-attachments/assets/3689a5c0-57b8-4dec-9531-c688f9b4a2cc" />
Stars (Ngôi sao): Bán chạy + Lãi cao (Ví dụ: Thai Chicken).
Workhorses (Ngựa thồ): Bán chạy nhưng lãi thấp (Do nguyên liệu đắt như Prosciutto).
Puzzles (Thách thức): Lãi cao nhưng bán chậm.
Dogs (Sản phẩm kém): Bán chậm + Lãi thấp.
**Hành động đề xuất: "Cần đẩy mạnh Marketing cho nhóm Stars và xem xét thay đổi nhà cung cấp hoặc điều chỉnh giá cho nhóm Workhorses."
## Điểm hòa vốn
<img width="1712" height="946" alt="image" src="https://github.com/user-attachments/assets/78f8706b-d64f-4772-b356-c725696c1c2a" />
Insight: "Với mức chi phí mặt bằng và vận hành hiện tại, cửa hàng cần đạt mức doanh thu tối thiểu là 25500$/tháng (tương đương khoảng 45 cái bánh/ngày) mới bắt đầu có lợi nhuận ròng (Net Profit)."

Hành động đề xuất: Nếu doanh thu hiện tại đang sát nút điểm hòa vốn, cần tăng giá trị trung bình trên mỗi đơn hàng (Up-selling combo nước/khoai tây chiên)

## Kết luận
Dự án cho thấy việc chỉ nhìn vào Doanh thu là một cái bẫy tài chính. Chỉ khi bóc tách được Chi phí nguyên liệu và Chi phí vận hành, cửa hàng mới thực sự kiểm soát được lợi nhuận bền vững. Kết quả cho thấy cửa hàng nên tập trung tối ưu hóa nhóm sản phẩm có biên lợi nhuận cao thay vì chỉ chạy theo doanh số của các dòng Pizza cao cấp nhưng chi phí quá lớn.
## 6. ⚙️ Hướng dẫn xem dự án
1. Clone dự án về máy.
2.SQL Sever : https://github.com/thangthai25/Pizza-Business-Analysis-Project/blob/main/analysis.sql
