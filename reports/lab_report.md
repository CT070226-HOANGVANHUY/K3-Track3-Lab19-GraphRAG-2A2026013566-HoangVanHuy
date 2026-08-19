# Báo cáo thực hành — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Hoàng Văn Huy
**Ngày:** 19/08/2026

## Kết quả thực thi

Pipeline live đã stream đủ 1.000 dòng từ HackerNoon vào `data/hf_live_1000.csv`. Sau lọc nội dung có 528 bài usable và 528 chunks; production extraction chạy trên 40 chunks đại diện để kiểm soát rate-limit. Xah API và Neo4j AuraDB đều kết nối thành công; graph live có 15 nodes, 8 edges và 0 edge thiếu `source_chunk_id`, `published_date`, `evidence` hoặc `confidence`.

Pipeline local reproducible cũng chạy thành công trên golden fixture: 50 chunks, 42 triples, 11 dòng entity-resolution audit và 0 edge thiếu provenance. Super-node test tạo node degree 120 và xác nhận chỉ lấy 50 cạnh. Hai file benchmark có đủ 50 câu và được xuất trong `outputs/`.

Top degree: ServiceNow (20), Aeris (13), NVIDIA (10). GraphRAG cải thiện rõ nhóm multi-hop: comprehensiveness 2.0 → 5.0, faithfulness 2.0 → 5.0 và reasoning 2.0 → 5.0. Flat RAG nhanh hơn (0.018s so với 0.031s) và dùng ít token hơn.

## Thuyết minh và reflection

- [Technical defense](technical_defense.md): 10 câu trả lời kiến trúc, threshold, super-node, trade-off và scale.
- [Failure analysis](failure_analysis.md): 2 ca lỗi cùng root cause và mitigation.
- [Reflection](reflection_HoangVanHuy.md): mapping module, debugging và action plan đồ án.

## Giới hạn môi trường

Dataset live hiện chứa nhiều bản ghi chỉ có phần `description`, nên sau lọc tối thiểu 80 ký tự còn 528 bài usable. Vì chi phí và thời gian gọi LLM, production extraction giới hạn ở 40 chunks đại diện; benchmark 50 câu trong `outputs/` là benchmark reproducible trên golden fixture, không gán nhãn là số đo toàn corpus live. Khi cần benchmark live đầy đủ, tăng `EXTRACTION_MAX_CHUNKS` và chạy lại với quota API phù hợp.
