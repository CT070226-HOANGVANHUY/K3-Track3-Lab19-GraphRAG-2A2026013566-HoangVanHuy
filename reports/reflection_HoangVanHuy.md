# Reflection & Action Plan — Hoàng Văn Huy

## 1. Mapping bài giảng vào code

| Khái niệm | Module | Hàm/artefact | Bài học |
|---|---|---|---|
| Conservative coreference | M1 | `conservative_coref()` / `resolve_coref_batch()` | Không rewrite khi antecedent mơ hồ. |
| Schema và allowlist | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `extract_fixture_triples()` | Precision trước recall để bảo vệ graph. |
| Bulk Cypher | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND` + batch giúp giảm round-trip. |
| Entity resolution | M3 | `resolve_entities()`, `canonicalize()` | Manual alias + ANN/lexical guard + audit. |
| Super-node cap | M4 | `retrieve_graph_context()`, `retrieve_graph()` | Degree cap và global cap bảo vệ context. |
| LLM-as-a-Judge | M5 | `judge_answer()`, `run_evaluation()` | Chấm riêng completeness, faithfulness và reasoning. |

## 2. Debugging và bài học

Lỗi khó nhất là pipeline ban đầu phụ thuộc đồng thời vào HF token, LLM key và Neo4j, đồng thời schema live dùng `description` thay vì `text`. Mình bổ sung mapping schema, giới hạn live ở 1.000 dòng, tách runner deterministic dùng golden fixture, viết `run_live_evaluation.py` để gọi Xah thật cho generation/Judge và kiểm tra provenance trước khi kết luận. Actual Judge cũng phát hiện mismatch quan trọng: Golden evidence thuộc 5.000 dòng nhưng graph live mới ingest 1.000 dòng.

## 3. Action plan đồ án

**Bài toán:** trợ lý hỏi đáp tri thức cho tài liệu dự án/phần mềm.

**Vì sao cần GraphRAG:** câu hỏi thường nối requirement → module → owner → release/incident qua nhiều tài liệu; Flat RAG phù hợp tra cứu đoạn văn đơn, còn GraphRAG phù hợp truy vết quan hệ và lịch sử thay đổi.

**Node:** `Project`, `Requirement`, `Service`, `Component`, `Person`, `Release`, `Incident`, `Document`.

**Relation:** `OWNS`, `DEPENDS_ON`, `IMPLEMENTS`, `AFFECTS`, `FIXED_IN`, `MENTIONED_IN`, `BLOCKS`.

**Entity resolution:** ID hệ thống là khóa chính; alias map cho mã service; ANN candidate chỉ là bước đề xuất, lexical/type guard và human review xử lý match rủi ro. Mỗi edge bắt buộc document/chunk/date/evidence.

**Super-node:** không expand trực tiếp node như `Project` hoặc `Person` có degree cao; lọc theo tenant, thời gian và relation type, lấy top-N gần nhất, community summary và global edge cap.

## Tự đánh giá

| Tiêu chí | Điểm | Ghi chú |
|---|---:|---|
| Hiểu GraphRAG | 4/5 | Nắm rõ graph giúp nối multi-hop và provenance. |
| Kiểm soát AI Agent | 4/5 | Tách fixture/reality, từ chối O(N²) và full-corpus LLM. |
| Chất lượng graph | 4/5 | Live Neo4j có 8 edge, provenance đầy đủ; cần mở rộng corpus để bao phủ toàn bộ Golden evidence. |
| Debug và phân tích | 4/5 | Có audit, policy test và failure analysis. |
