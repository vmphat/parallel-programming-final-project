# Đồ án cuối kỳ - Lập trình song song - 21KHMT

## Sinh viên thực hiện: Nhóm 05 - Thần y xứ Quảng

| STT | MSSV     | Họ và tên         |
| :-: | -------- | ----------------- |
|  1  | 21127084 | Lê Xuân Kiên      |
|  2  | 21127108 | Đặng Hà Nhật Minh |
|  3  | 21127739 | Vũ Minh Phát      |

## Link đến playlist video trình bày kết quả của đồ án

- Link đến playlist video trình bày kết quả của đồ án: https://www.youtube.com/playlist?list=PLY_ov3yMCYsdSNYektjNKpKpVDOnbqabk
- Trong playlist sẽ có 2 video:
  - Video có tiêu đề kết thúc với "(Presentation)": Là video thuyết trình chi tiết file báo cáo của nhóm.
  - Video có tiêu đề kết thúc với "(Live - Demo)": Là video thể hiện quá trình chạy thực tế của chương trình.

## Cách tổ chức thư mục chứa mã nguồn ("./src")

- Thư mục "**`src`**" chứa tất cả mã nguồn của đồ án:
  - Thư mục con "All-in-one": chứa mã nguồn được sử dụng trong quá trình phát triển tất cả 6 mô hình (tuần tự và song song).
  - Thư mục con "Host-V1": chứa mã nguồn được sử dụng trong quá trình phát triển mô hình tuần tự lần 1.
  - Thư mục con "Host-V2": chứa mã nguồn được sử dụng trong quá trình phát triển mô hình tuần tự lần 2.
  - Thư mục con "Parallel-V1": chứa mã nguồn được sử dụng trong quá trình phát triển mô hình song song lần 1.
  - Thư mục con "Parallel-V2": chứa mã nguồn được sử dụng trong quá trình phát triển mô hình song song lần 2.
  - Thư mục con "Parallel-V3": chứa mã nguồn được sử dụng trong quá trình phát triển mô hình song song lần 3.
  - Thư mục con "Parallel-V4": chứa mã nguồn được sử dụng trong quá trình phát triển mô hình song song lần 4.
  - Thư mục con "Test-kernel-functions": chứa 1 file notebook và các file code để kiểm tra tính đúng đắn của các hàm kernel được sử dụng trong quá trình phát triển mô hình song song.
  - Thư mục con "Live-Demo": chứa 1 file notebook và các file code để chạy chương trình minh họa trong video "Live - Demo".

## Kế hoạch thực hiện đồ án và phân công công việc

- Tài liệu về kế hoạch thực hiện đồ án của nhóm 05 và bảng phân công công việc chi tiết cho mỗi thành viên đã được đính kèm trong file "**`Team-Plan-and-Work-Distribution.xlsx`**" của bài nộp.

## Hướng dẫn chạy chương trình trong file "Report.ipynb"

1. Đầu tiên, ta sẽ truy cập đến [đường dẫn này](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) để tải 4 file chứa dữ liệu huấn luyện và kiểm tra của bộ dữ liệu Fashion-MNIST.

2. Mở file "Report.ipynb" được đính kèm trong bài nộp bằng Google Colab.

3. Sau khi mở file notebook bằng Google Colab, ta sẽ kết nối thời gian chạy của notebook với GPU bằng cách chọn "Runtime" -> "Change runtime type" -> chọn "GPU" -> "Save".

4. Tiếp theo, ta sẽ upload 4 file chứa dữ liệu huấn luyện và kiểm tra đã tải ở bước 1 lên Google Colab.

5. Mở thư mục bài nộp, chọn thư mục "src" -> "All-in-one", sau đó upload toàn bộ các file code trong thư mục "All-in-one" lên Google Colab.

6. Cuối cùng, ta có thể chạy tất cả các cell trong file "Report.ipynb" để xem kết quả của chương trình. Chọn "Runtime" -> "Run all" để chạy tất cả các cell trong notebook.
