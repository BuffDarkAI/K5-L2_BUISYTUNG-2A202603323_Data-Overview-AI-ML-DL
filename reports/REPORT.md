# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`="468" ; `class_name`="cab" ; `rank`=1 ; `score`="0.510915" ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
 - Record này nói về cả bức ảnh, không phải một vật thể riêng biệt trong ảnh,vì mô hình chỉ đưa ra nhãn đại diện cho một bức ảnh chứ không xác định hay vẽ bounding box bao quanh vật thể
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
 - danh sách class list được đi kèm taxonomy/checkpoint chứ không phải model tự nghĩ ra  và khi dự đoán dựa vào các lớp đã được huấn luyện và đưa ra kết quả phù hợp nhất cùng điểm confidence  
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
 - vì để tránh nhầm lẫn dữ liệu vì ghi đầy đủ ba thông tin giúp dữ liệu rõ ràng, chính xác, dễ quản lý, tra cứu và sử dụng lại trong quá trình huấn luyện mô hình AI
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
 - một mã duy nhất hoặc đối tượng chiếm ưu thế(các đối tượng được train chính) trong ảnh chứ không gán nhanx cho vật thể phụ trong nahr 
6. Vì sao không được lấy `score` của model làm ground truth?
 - vì score là mức độ tự tin của mô hình, hỗ trợ các trường hợp kiểm tra thủ công và đnahs gia chất lượng mô hình, còn ground truth phải được đánh gía độc lập
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`=  0.769676 ; `bbox_xyxy`=[
      0.08,
      256.79,
      18.39,
      313.12
    ],
     ; `bbox_width`= 18.32 ; `bbox_height`=  56.33
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 - góc trên bên trái của box tại x =0.08 , y= 256.79, góc dưới phải của box tại x =18.39, y= 313.12 và box rộng 18.32 cao  56.33
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
 - ngưỡng 0.2 có 17 vật thể và ngưỡng 0.6 có 6 vật thể 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 - cần kiểm tra nhiều bounding box hơn, xem nhãn label có chính xác không 
 - loại bỏ false positives 
 - xử lý các hợp trùng lặp 
 - tăng thời gian review
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
 - Vẽ bounding box ôm sát toàn bộ phần nhìn thấy của đối tượng, sao cho đối tượng chiếm phần lớn diện tích hộp, không cắt mất bộ phận của đối tượng và không lấy quá nhiều nền xung quanh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
 - Đưa thành quy tắc trong guideline, không cần escalation
 - Tình huống vật thể bị che khuất hoặc cắt mép ảnh là rất phổ biến trong các bộ dữ liệu object detection
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-004 ; `class_name`= potted plant ; `score`= 0.632235 ; số điểm polygon=`polygon_point_count`= 100 ; 3–5 điểm đầu của `polygon_xy`=
 [
        1.0,
        2.0
      ],
      [
        1.0,
        148.0
      ],
      [
        3.0,
        148.0
      ],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 - box chỉ bao quanh bằng hình chữ nhật còn polygon nó bao quanh vật thể bằng nhiều điểm của đường viền vật thể
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
 - instance_id dùng để Xác định một đối tượng cụ thể trong ảnh, mỗi đối tượng được phát hiện sẽ có một ID riêng giúp phân biệt các đối tượng riêng biệt, kể cả khi chúng thuộc cùng một class
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
 - Người gán nhãn phải vẽ polygon ôm sát đường viền thực tế của đối tượng cần phân đoạn
 chỉ bao phủ các pixel thuộc đối tượng, không bao gồm nền hoặc các vật thể khác. Đường biên phải liên tục, bám theo hình dạng quan sát được của đối tượng, không tự ý vẽ thêm phần bị che khuất. Nếu đối tượng bị che khuất một phần, chỉ vẽ phần thực sự nhìn thấy. Mỗi đối tượng riêng biệt phải có một mask riêng, không gộp nhiều đối tượng cùng lớp thành một mask.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
 - dùng guideline Đây là tình huống có thể xử lý theo nguyên tắc vẽ mask đã quy định: bám sát đường viền của phần đối tượng nhìn thấy được, Không tự ý suy đoán phần bị che khuất và Chỉ đánh dấu vùng thực sự nhìn thấy trong ảnh.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một ảnh có một nhãn class duy nhất, ví dụ: traffic, kitchen, dining.  | Ảnh traffic có nhiều loại phương tiện. Model dự đoán nhiều class như cab, minibus, police_van|  Xác định chủ đề chính của ảnh theo guideline |  Kiểm tra nhãn có phản ánh đúng chủ đề chính không, có tuân thủ taxonomy và quy tắc một ảnh – một nhãn không.|
| Phát hiện vật thể | Mỗi vật thể có một nhãn class và một bounding box (x_min, y_min, x_max, y_max) |Nhiều xe nằm sát nhau, một số xe bị che khuất. Bounding box có thể bao gồm nền hoặc thiếu một phần vật thể  |Vẽ bounding box riêng cho từng vật thể nhìn thấy, chọn đúng class và không gộp các vật thể riêng biệt.  |Kiểm tra class, số lượng vật thể, vị trí và độ chính xác của bounding box; chú ý vật thể bị bỏ sót hoặc box chồng lấn sai. |
| Instance segmentation | Mỗi đối tượng có một instance_id, class_id và polygon mask riêng. | JSON kitchen-004 có class_id = 58, class_name = potted plant, score = 0.632235 |  Vẽ mask theo đường biên thực tế của từng đối tượng; mỗi instance có ID riêng. Chỉ đánh dấu phần nhìn thấy, không tự suy đoán phần bị che.|Kiểm tra mask có ôm sát đối tượng không, polygon có hợp lệ không, instance ID có duy nhất không và class có khớp với đối tượng không.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
 - Không đưa họ tên, MSSV, email, số điện thoại hoặc bất kỳ thông tin cá nhân nhạy cảm nào vào dữ liệu, JSON, hình ảnh, báo cáo và repository công khai của bài lab.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
 - Tôi sẽ dừng xử lý dữ liệu và báo ngay cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn, không tự ý xử lý, chỉnh sửa hoặc chia sẻ dữ liệu ngoài phạm vi cho phép.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
