# Failure Analysis — GraphRAG vs Flat RAG

## Case A — Flat RAG miss trên câu hỏi multi-hop

**Question:** G5000-01 — reconstruct Aeris–Ericsson transaction và quy mô IoT footprint.

**Observed failure:** Flat retrieval có thể lấy chunk nói về việc mua IoT Accelerator/Connected Vehicle Cloud nhưng không lấy chunk muộn hơn chứa 100 million devices, 9,000 enterprises và 190 countries. Câu trả lời vì vậy chỉ đúng một nửa.

**Root cause:** semantic top-k tối ưu độ tương đồng của từng chunk, không có cơ chế nối entity giữa các report và không biết rằng cùng một transaction đã đổi trạng thái theo thời gian.

**GraphRAG mitigation:** seed `Aeris`, `Ericsson`, `IoT Accelerator`, `Connected Vehicle Cloud`; BFS nối các node, sau đó linearize edge với `source_chunk_id`, date và evidence. Hybrid context vẫn giữ vector chunks để bổ sung số liệu.

## Case B — GraphRAG khó với temporal state

**Question:** G5000-02 — planned transfer hay completed acquisition?

**Observed failure:** schema relation hiện tại có `ACQUIRED` nhưng chưa có event-state field. Nếu extractor chuẩn hóa cả “was to acquire” và “has acquired” thành cùng một relation, graph làm mất diễn tiến planned → completed.

**Root cause:** relation allowlist chưa biểu diễn modality/status; hai report có cùng entity pair nhưng khác thời điểm và khác event state.

**Mitigation:** thêm `status` (`PLANNED`, `COMPLETED`), `valid_from`, `valid_to`, giữ nguyên evidence; khi conflict thì sort theo `published_date`, không overwrite lịch sử. Retrieval nhận temporal intent của query; nếu seed không match thì dùng fuzzy fallback và vector retrieval.

## Kiểm tra failure-mode

- Provenance: 42/42 fixture edges có chunk/date; invalid count = 0.
- Entity audit: 10 quyết định merge/reject được xuất trong `outputs/entity_resolution_audit.csv`.
- Super-node: synthetic node degree 120 chỉ fetch 50 edge, test pass.
