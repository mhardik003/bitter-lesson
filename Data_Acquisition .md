## **Data Acquisition Report: Indian Legal Corpus on Contracts and Agreements**

### **Phase 1 — Initial Collection via `wget`**

At the project’s outset, we needed a small seed corpus of publicly available Indian legal documents.
We used `wget`, a command-line utility for recursive file downloads, to fetch accessible judgment and order PDFs directly from government domains—particularly from the **Supreme Court of India ([https://www.sci.gov.in/](https://www.sci.gov.in/))** and related portals.

**Objective:**
Build an initial reference dataset of authoritative judgments to analyze citation patterns and contract-law terminology.

**Method:**

```bash
# Rail
wget "https://www.iricen.gov.in/iricen/Works_Manuals/GCC-2022-ACS6.pdf"
wget "https://indianrailways.gov.in/railwayboard/view_section.jsp?id=0%2C1%2C304%2C366%2C526%2C2624&lang=0"        # index page (bookmark for updates)
wget "https://dfccil.com/upload/GCC_2022_4DXM.pdf"

# Ports PPP
wget "https://shipmin.gov.in/sites/default/files/Model%20Concession%20Agreement%2C%202021%20with%20Guiding%20Note_compressed_0.pdf"
wget "https://www.pppinindia.gov.in/model_concession_agreement"  # index listing with port MCA link

# Pipelines / Gas transportation
wget "https://www.pngrb.gov.in/pdf/public-notice/DGTA24062021.pdf"          # PNGRB Gas Transportation Agreement format
wget "https://gailebank.gail.co.in/goga/NewApplication/pdf/Approved%20GTA.pdf" # GAIL approved GTA template

# PSU GCCs — Services/Goods (for variation across CPSEs)
wget "https://tenders.bhel.com/sites/default/files/GCC_REV01-2025-01-02-08%3A58%3A10.pdf"           # BHEL GCC (Services)
wget "https://pem.bhel.com/Documents/GCC/GCC-Rev-04.pdf"                                            # BHEL PEM GCC
wget "https://gailonline.com/pdf/gcc/GCC-GoodsRev1.pdf"                                             # GAIL GCC Goods Rev.1 (Apr 2022)
wget "https://gailonline.com/pdf/gcc/GCCCONSULTANCYSERVICES.pdf"                                    # GAIL GCC Consultancy Services
wget "https://gailgaspdfdownloads.s3.ap-south-1.amazonaws.com/General-Conditions-of-Contract-Works%28GCC-Works%29-English-version.pdf"  # GAIL Gas GCC Works
wget "https://iocl.com/uploads/PartB03022025.pdf"                                                   # IOCL tender Part B incl. GCC sections

```


This provided a handful of **authentic judgments** in PDF form, which served as initial test data for text-extraction pipelines.

---

### **Phase 2 — Manual Case Downloads from `sci.gov.in`**

Because the Supreme Court website doesn’t expose a structured API and its search results often require form submissions, we **manually curated** an extended set of contract-related cases.

**Approach:**

1. Searched manually using the SCI site’s “Judgments” page for terms such as
   *“contract act”*, *“agreement”*, *“Article 299”*, *“specific performance”*.
2. Downloaded PDFs of judgments individually.
3. Renamed them systematically using the citation or date (e.g.,
   `2023_SC_Contract_Article299.pdf`).

This stage produced a verified, high-quality subset of landmark cases, useful both for model calibration and for benchmarking later automated scraping results.

---

### **Phase 3 — Automated Crawling of Indian Kanoon**

Once the foundational data were secured, we expanded coverage using **Indian Kanoon ([https://indiankanoon.org)**—a](https://indiankanoon.org%29**—a) comprehensive aggregator of Indian court and tribunal judgments.

#### **3.1 Scraping Strategy Overview**

We built a Python-based crawler in two components:

1. **`indian_kanoon_scraper.py`**
   Queries the public search interface, paginates through results, and records judgment links.
2. **`ik_pdf_downloader.py`**
   Reads those links, fetches each case page, follows the “Get this document in PDF” link, and saves the resulting files.

#### **3.2 Search-Query Design**

To ensure breadth and relevance, queries combined statutory anchors and contextual terms.
Example patterns:

```
Section 10 Indian Contract Act validity
Section 23 Indian Contract Act public policy
Article 299 Constitution of India government contract
power purchase agreement tariff CERC
```

Each phrase was tested in a browser first to confirm results existed; then the scraper automatically iterated through **up to 100 pages per query**.

#### **3.3 Scraper Architecture**

* **Session Handling:** Uses `requests.Session()` with realistic headers to avoid lightweight responses or throttling.
* **Pagination:** Constructs URLs with `formInput` and `pagenum` parameters.
* **Link Extraction:** Parses HTML via `BeautifulSoup`, capturing all canonical `/doc/<id>/` links.
* **De-duplication:** Maintains a global set of seen URLs; appends new links to `links.txt` only once.
* **Politeness:** Configurable `--delay` (1–2 s default) between requests to respect server load.
* **Resilience:** Gracefully skips missing or empty pages.

#### **3.4 PDF Downloader Logic**

* Reads `links.txt` generated above.
* Checks `downloaded.txt` to skip already processed URLs.
* For each new link:

  1. Fetches the judgment page.
  2. Locates the *“Get this document in PDF”* anchor.
  3. Downloads and saves the file to the `pdfs/` directory.
  4. Logs successful URLs in `downloaded.txt`; any failures in `failures.txt`.
* Filenames include both the document ID and judgment title for traceability.
* Partial downloads are written to `.part` files and atomically renamed after success to prevent corruption.

#### **3.5 Logging and Resumption**

Both scripts are **append-only** and idempotent:

* `links.txt` — all discovered case URLs
* `downloaded.txt` — confirmed successful PDF downloads
* `failures.txt` — problematic URLs for later retry
  This enables pausing and resuming long crawls without losing progress.

---

### **Phase 4 — Corpus Assembly and Next Steps**

The combined pipeline now yields:

1. A **clean corpus of judgment PDFs** covering contract law under the Indian Contract Act (1872), constitutional contract clauses, and industry-specific disputes.
2. Machine-readable logs for reproducibility and deduplication.
3. A base for downstream NLP preprocessing (text extraction, section tagging, and embedding generation).

Planned refinements include:

* Metadata parsing (court, bench, citation, date).
* Optical Character Recognition fallback for scanned PDFs.
* Periodic incremental updates via new search terms.

---

### **Summary**

| Phase | Source               | Method                         | Outcome                         |
| ----- | -------------------- | ------------------------------ | ------------------------------- |
| 1     | Public govt. domains | `wget` (filtered PDFs)         | Initial seed corpus             |
| 2     | SCI portal           | Manual curated downloads       | Verified Supreme Court cases    |
| 3     | Indian Kanoon        | Automated scraper + downloader | Large-scale corpus of judgments |
| 4     | Consolidation        | Deduplication + metadata       | Ready dataset for analysis      |

