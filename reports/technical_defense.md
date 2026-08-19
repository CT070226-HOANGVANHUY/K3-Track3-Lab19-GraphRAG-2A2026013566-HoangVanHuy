# Technical Defense — Lab 19

**Học viên:** Hoàng Văn Huy
**Ngày:** 19/08/2026
**Chế độ chạy:** live ingestion 1.000 dòng + local fixture benchmark; không hard-code credentials.

## 1. Coreference Resolution

Pipeline dùng quy tắc conservative: chỉ thay đại từ khi antecedent xuất hiện rõ trong cùng chunk. Với fixture hiện tại, các mention mơ hồ được giữ nguyên và ghi vào `unresolved_mentions`; không tự suy diễn để tạo edge. Đây là lựa chọn an toàn vì một coreference sai có thể chuyển quan hệ của công ty A sang công ty B và tạo false edge. Khi chạy dataset đầy đủ, cần spot-check các câu có “the company/it” trước khi bật rewrite tự động.

## 2. Entity-resolution threshold

Ngưỡng vector được giữ ở `cosine >= 0.90`; lexical guard sau đó kiểm tra hậu tố doanh nghiệp và `SequenceMatcher >= 0.72`. Bảng `outputs/entity_resolution_audit.csv` có 11 dòng audit, gồm `MERGE_MANUAL`, `MERGE_VECTOR` và `REJECT_GUARD`.

Một cặp cố ý bị chặn là `Sam Altman` và `Steve Altman` với similarity 0.91: cùng họ không đủ bằng chứng để gộp hai người. Tương tự `Apple Watch`/`Apple` và `Apple Music`/`Apple` bị chặn vì sản phẩm/dịch vụ không phải cùng node Company.

## 3. Union-Find và audit

Các alias chắc chắn như `ServiceNow Inc` → `ServiceNow` dùng `MERGE_MANUAL`. Các cặp còn lại chỉ được nối khi qua lexical guard; Union-Find giúp việc gộp có tính bắc cầu mà không cần tạo tất cả pairwise edges trong graph. Mọi quyết định đều được ghi lại với tên trái/phải, similarity và decision để audit ngược.

## 4. Topology và super-node

Top 3 thực thể trong fixture:

| Hạng | Entity | Type | Degree |
|---:|---|---|---:|
| 1 | ServiceNow | Company | 20 |
| 2 | Aeris | Company | 13 |
| 3 | NVIDIA | Company | 10 |

Nếu degree > 100, BFS chỉ lấy 50 cạnh mới nhất và toàn context tối đa 250 cạnh/14.000 ký tự. Chính sách này chặn token explosion và ưu tiên trạng thái mới. Rủi ro là câu hỏi lịch sử có thể bị cắt mất cạnh cũ; mitigation là temporal filter theo năm hỏi, community summary hoặc hop-3 fallback.

## 5. Bulk ingestion và provenance

Notebook giữ Cypher `UNWIND $rows AS row`, batch 1.000 records, constraint unique trên `Entity.id` và index `name_norm`. Triple fixture có 42 edge; mỗi edge có `source_chunk_id`, `published_date`, `evidence`, `confidence`. Live Neo4j hiện có 8 edge với đủ 4 thuộc tính bắt buộc; sanity check trả `invalid_provenance_edges = 0`.

## 6. Flat RAG vs GraphRAG

Actual LLM Judge bằng Xah đã chấm đủ 50 câu Golden trên context live:

| Metric | Flat RAG | GraphRAG | Delta |
|---|---:|---:|---:|
| Comprehensiveness | 1.620 | 1.260 | -0.360 |
| Faithfulness | 2.380 | 2.260 | -0.120 |
| Multi-hop reasoning | 1.740 | 1.300 | -0.440 |
| Latency (s) | 1.198 | 1.109 | -0.089 |
| Token usage | 697.080 | 695.380 | -1.700 |

GraphRAG thấp hơn trong lần chạy live này vì graph chỉ được dựng từ 1.000 dòng, còn Golden Dataset ghi rõ evidence thuộc phạm vi 5.000 dòng. Đây là mismatch về corpus coverage, không phải bằng chứng GraphRAG luôn kém. Khi mở rộng ingestion/extraction tới đúng phạm vi Golden, cần chạy lại benchmark để đo lợi thế multi-hop công bằng.

## 7. Ca lỗi Flat RAG

Với G5000-01, các thông tin “IoT Accelerator/Connected Vehicle Cloud” và “100 million devices/9,000 enterprises/190 countries” nằm ngoài phạm vi graph live 1.000 dòng. Flat RAG vẫn có thể lấy được một phần nhờ vector similarity; GraphRAG không có đủ seed/evidence trong Neo4j nên actual Judge chấm thấp. Root cause là corpus coverage mismatch; mitigation là ingest đúng 5.000 dòng hoặc đánh dấu câu hỏi ngoài phạm vi.

## 8. Ca lỗi GraphRAG

GraphRAG có thể thất bại khi seed extraction không nhận diện alias, extraction bỏ sót quan hệ hoặc super-node temporal cap loại cạnh lịch sử. Ví dụ câu hỏi về trạng thái giao dịch Aeris–Ericsson có thể không phân biệt “was to acquire” với “has acquired” nếu schema chỉ lưu `ACQUIRED` mà không lưu event status. Khắc phục: thêm thuộc tính `status`, `valid_from/valid_to`, evidence-level conflict resolution và fallback vector.

## 9. Trade-off và scale 350MB

Flat RAG có indexing đơn giản, latency/token thấp nhưng thiếu đường nối giữa các document. GraphRAG tốn extraction, entity resolution và traversal nhưng phù hợp câu hỏi nhiều bước, temporal và provenance. Khi scale 350MB, bottleneck đầu tiên là LLM extraction và embedding/ANN, không phải Cypher `UNWIND`. Giải pháp là queue async theo batch, cache theo hash, retry idempotent, HNSW/FAISS, blocking theo type/name và community partitioning.

## 10. Kiểm soát AI Coding Agent

Quyết định từ chối là chạy pairwise similarity toàn bộ entity mentions (`O(N²)`) và gửi toàn bộ 350MB qua LLM. Cả hai đều tạo nguy cơ OOM/rate-limit. Thay vào đó dùng ANN top-k, lexical guard, batch extraction, scale guard và checkpoint CSV. Actual evaluator `run_live_evaluation.py` dùng Xah cho generation và Judge; runner local vẫn được giữ riêng để kiểm thử contract reproducible.
