# Báo cáo thực hành — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Hoàng Văn Huy
**Ngày:** 19/08/2026

## Kết quả thực thi

Pipeline live đã stream đủ 1.000 dòng từ HackerNoon vào `data/hf_live_1000.csv`. Sau lọc nội dung có 528 bài usable và 528 chunks; production extraction chạy trên 40 chunks đại diện để kiểm soát rate-limit. Xah API và Neo4j AuraDB đều kết nối thành công; graph live có 15 nodes, 8 edges và 0 edge thiếu `source_chunk_id`, `published_date`, `evidence` hoặc `confidence`.

Pipeline local reproducible cũng chạy thành công trên golden fixture: 50 chunks, 42 triples, 11 dòng entity-resolution audit và 0 edge thiếu provenance. Super-node test tạo node degree 120 và xác nhận chỉ lấy 50 cạnh. Hai file benchmark có đủ 50 câu và được xuất trong `outputs/`.

LLM Judge thật bằng Xah đã chấm đủ 50 câu: điểm tổng hợp Flat RAG là 1.913/5, GraphRAG là 1.607/5. Trong live corpus 1.000 dòng, GraphRAG chưa thắng vì Golden evidence thuộc phạm vi 5.000 dòng; graph hiện thiếu phần lớn evidence cần cho các câu hỏi khó. Latency trung bình là 1.198s Flat và 1.109s Graph; token trung bình là 697.08 và 695.38.

## Thuyết minh và reflection

- [Technical defense](technical_defense.md): 10 câu trả lời kiến trúc, threshold, super-node, trade-off và scale.
- [Failure analysis](failure_analysis.md): 2 ca lỗi cùng root cause và mitigation.
- [Reflection](reflection_HoangVanHuy.md): mapping module, debugging và action plan đồ án.

## Giới hạn môi trường

Dataset live hiện chứa nhiều bản ghi chỉ có phần `description`, nên sau lọc tối thiểu 80 ký tự còn 528 bài usable. Production graph dùng 40 chunks đại diện để kiểm soát rate-limit; actual benchmark 50 câu trong `outputs/` đã dùng Xah cho retrieval generation và LLM Judge. Vì Golden Dataset tham chiếu phạm vi 5.000 dòng còn graph live mới ingest 1.000 dòng, kết quả thấp của GraphRAG là một failure mode có nguyên nhân rõ ràng.
