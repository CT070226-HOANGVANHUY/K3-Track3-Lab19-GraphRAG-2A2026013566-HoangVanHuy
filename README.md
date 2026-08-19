# Lab 19: Production-Grade GraphRAG vs Flat RAG

**AICB-K34 · Ngày 19 · Track 3: GraphRAG**  
**Thời lượng:** 2h implement + 30 phút reflection & thuyết minh kỹ thuật  
**Môi trường:** Google Colab (T4 GPU khuyến nghị) / Jupyter Notebook + Neo4j AuraDB  
**Dữ liệu:** HackerNoon Tech Company News Data Dump (`HackerNoon/tech-company-news-data-dump`)  
**Công cụ:** Học viên được dùng AI Coding Agent, nhưng phải tự thiết kế, kiểm thử, audit dữ liệu và bảo vệ kiến trúc.

---

## 🎯 Tổng quan

Bài tập lab toàn diện so sánh **Flat RAG (Vector Search)** với **Production GraphRAG (Knowledge Graph + Hybrid Retrieval)**:

```
Stream Dataset → Dedup & Chunking → Coreference Resolution
                                           │
   ┌───────────────────────────────────────┴───────────────────────────────────────┐
   ▼                                                                               ▼
[Flat RAG Index]                                                           [NER + RE Extraction]
Vector Embeddings + FAISS FlatIP                                                   │
   │                                                                               ▼
   │                                                                      [Entity Resolution]
   │                                                                   Vector ANN + Lexical Guard
   │                                                                               │
   │                                                                               ▼
   │                                                                    [Neo4j Bulk Insert]
   │                                                                   UNWIND + Edge Provenance
   │                                                                               │
   │                                                                               ▼
   │                                                                      [Graph Traversal]
   │                                                                  BFS + Super-node Mitigation
   │                                                                               │
   └───────────────────────────────────────┬───────────────────────────────────────┘
                                           ▼
                            [Hybrid Context & Generation]
                                           ▼
                        [Golden Evaluation & LLM-as-a-Judge]
                    (Factoid · Multi-hop · Cross-doc Reasoning)
```

Xem **[ASSIGNMENT.md](ASSIGNMENT.md)** để biết chi tiết từng module, yêu cầu kỹ thuật và 10 câu hỏi thuyết minh.  
Xem **[RUBRIC.md](RUBRIC.md)** để biết tiêu chí đánh giá và thang điểm (100 điểm + 10 bonus).

---

## 📋 Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| **Neo4j AuraDB** (hoặc Neo4j 5.x) | ✅ Có | Lưu trữ Knowledge Graph & Cypher traversal |
| **Python 3.10+ / Colab** | ✅ Có | Môi trường thực thi Notebook |
| `HF_TOKEN` | ✅ Có | Stream dataset từ Hugging Face (`HackerNoon`) |
| `GROQ_API_KEY` | ✅ Có | Coreference, NER+RE Extraction, Seed Extraction, Generator |
| `OPENAI_API_KEY` | ⚠️ Có thể thay thế | LLM-as-a-Judge (có thể cấu hình dùng Groq hoặc OpenAI) |

### Cấu hình Secrets (Colab Secrets hoặc `.env`)

Khai báo các biến môi trường sau:

```bash
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j

GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# Provider OpenAI-compatible theo mẫu trong ảnh
LLM_PROVIDER=xah
XAH_API_KEY=sk-...
XAH_BASE_URL=https://api.xah.io/v1
XAH_MODEL=mainnewnol/gpt-5.6-luna

JUDGE_PROVIDER=xah                 # 'xah', 'openai' hoặc 'groq'
JUDGE_MODEL=mainnewnol/gpt-5.6-luna
OPENAI_API_KEY=sk-...

HF_TOKEN=hf_...                     # Hugging Face User Access Token
```

> [!WARNING]
> **Tuyệt đối không hard-code API Key hoặc mật khẩu Neo4j** vào notebook khi nộp bài.

Provider Xah dùng API chuẩn tương thích OpenAI: base URL là `https://api.xah.io/v1`,
route chat là `/chat/completions`, và model mặc định là `mainnewnol/gpt-5.6-luna`.
Notebook sẽ đọc key từ `XAH_API_KEY` (hoặc alias `XAI_API_KEY`), không ghi key vào mã nguồn.

---

## ⚡ Quick Start

### Cách 1: Chạy trực tiếp trên Google Colab (Khuyến nghị)
1. Mở file [`Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb`](Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb) trên Google Colab.
2. Thêm các secret keys vào tab **Secrets (biểu tượng chiếc khóa 🔑)** trên Colab.
3. Chạy từng section theo Timeline hướng dẫn.

### Cách 2: Chạy Local Notebook
```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Pre-download embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 3. Tạo file .env và điền API keys
cp .env.example .env

# 4. Khởi chạy Jupyter Lab / Notebook
jupyter lab Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb
```

### Chạy local không cần credentials

PowerShell trên Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe lab_runner.py
```

`lab_runner.py` dùng golden fixture có sẵn để kiểm thử reproducible preprocessing, triple extraction, entity audit, provenance, super-node policy và benchmark; kết quả được xuất vào `outputs/`. Khi có credentials thật, điền `.env` theo `.env.example` rồi chạy notebook để chuyển sang HF/LLM/Neo4j production path.

Để chạy actual generation + LLM-as-a-Judge trên đủ 50 câu Golden bằng Xah:

```powershell
.venv\Scripts\python.exe run_live_evaluation.py
```

Lệnh này dùng `hf_live_1000.csv`, gọi Xah thật cho Flat/Graph generation và Judge, sau đó ghi đè hai file benchmark trong `outputs/`.

---

## ⏳ Timeline (120 phút + 30 phút Thuyết minh)

| Thời gian | Module | Trọng tâm kỹ thuật |
|-----------|--------|-------------------|
| **0:00–0:15** | **Phần 1: Setup & Preprocessing** | Stream HF data, exact dedup, text chunking, conservative coreference resolution |
| **0:15–0:45** | **Phần 2: Triple Extraction & Neo4j Ingestion** | NER + RE với JSON mode, schema allowlist, Entity Resolution (Vector ANN + Lexical Guard), bulk insert `UNWIND` |
| **0:45–1:15** | **Phần 3: Flat RAG & Hybrid GraphRAG** | FAISS Flat RAG index, Seed extraction, BFS graph traversal, Super-node mitigation (degree > 100 → cap 50) |
| **1:15–1:45** | **Phần 4: Golden Eval & Benchmark** | Chạy 5+ Golden queries, LLM-as-a-Judge (1–5 scale), bảng so sánh Quality / Latency / Tokens |
| **1:45–2:00** | **Phần 5: Failure Modes & Bonus** | Super-node check, Entity audit log, Bonus Global Search & Self-Correction |
| **2:00–2:30** | **Reflection & Thuyết minh** | Trả lời 10 câu hỏi kỹ thuật + Lecture Mapping + Action Plan |

---

## 🛡️ Scale Guard (Quy tắc an toàn dữ liệu trong Lab)

Trong thời lượng 2 giờ, để tránh cạn kiệt rate limit hoặc tràn bộ nhớ:
- `LAB_MAX_ARTICLES = 1500` (Số bài báo tối đa)
- `LAB_MAX_CHUNKS = 3000` (Số chunk văn bản tối đa)
- `EXTRACTION_MAX_CHUNKS = 400` (Số chunk trích xuất đồ thị)
- `CHUNK_WORDS = 220`, `CHUNK_OVERLAP_WORDS = 40`

---

## 📂 Cấu trúc Repo

```
Day19-Track3-GraphRAG/
├── README.md                                             # Hướng dẫn tổng quan, scale guard, setup, timeline
├── ASSIGNMENT.md                                         # Đề bài chi tiết 5 modules & hướng dẫn thực hiện
├── RUBRIC.md                                             # Thang điểm chi tiết (100đ + 10 bonus)
├── .env.example                                          # Template biến môi trường
├── .gitignore                                            # Cấu hình bỏ qua file lớn & API keys
├── requirements.txt                                      # Thư viện Python cần thiết
├── Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb   # ★ File Notebook thực hành chính
│
├── data/                                                 # 📁 Chứa dữ liệu & Golden schema
│   └── golden_dataset.csv                                # Schema & 5 câu hỏi đánh giá mẫu (G01–G05)
│
├── outputs/                                              # 📁 File kết quả xuất tự động từ notebook (*.csv)
│   ├── graphrag_eval_results.csv                         # Chi tiết kết quả từng câu hỏi + điểm Judge
│   └── graphrag_vs_flatrag_summary.csv                   # Bảng so sánh tổng hợp Flat RAG vs GraphRAG
│
├── reports/                                              # 📁 Báo cáo hoàn chỉnh của học viên (Chỉ 1 file duy nhất)
│   └── lab_report.md                                     # ★ Thuyết minh kỹ thuật (10 câu) + Phân tích lỗi + Reflection
│
└── templates/                                            # 📁 Bản sao dự phòng gốc của mẫu báo cáo
    └── lab_report.md
```

---

## 🚀 Deliverables (Bài nộp)

Học viên commit và push lên GitHub cá nhân:
1. `Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb` (Notebook đã chạy đầy đủ output các cell).
2. `outputs/graphrag_eval_results.csv` và `outputs/graphrag_vs_flatrag_summary.csv`.
3. `reports/lab_report.md` (Điền đầy đủ 2 phần: Thuyết minh kỹ thuật & Suy ngẫm cá nhân).
