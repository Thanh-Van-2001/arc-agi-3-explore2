# Checklist nộp ARC-AGI-3 + Paper Track (tài khoản thnhnguyn2001)

Tất cả chất liệu đã sẵn trong `D:\arc_agi_3\`. Làm theo thứ tự.

## PHẦN A — Nộp ARC-AGI-3 (prediction, lấy điểm leaderboard)

1. **Vào** kaggle.com/competitions/arc-prize-2026-arc-agi-3 → **Join/Accept rules** (bắt buộc trước 26/10).
2. **Create Notebook** (Code → New Notebook, gắn competition này).
3. Panel phải **Add Input** → tìm & thêm dataset của competition (mount thành
   `/kaggle/input/competitions/arc-prize-2026-arc-agi-3`, path đã khớp sẵn trong code).
4. **Xóa cell mẫu**, mở `D:\arc_agi_3\kaggle_submission_notebook.py`, **copy TOÀN BỘ**, dán vào 1 cell.
5. **Settings (panel phải):**
   - Internet: **OFF**
   - Accelerator: **None** (CPU đủ; nếu muốn nhanh hơn có thể chọn nhưng không cần)
   - Persistence: không cần
6. **Save Version → Save & Run All (Commit)**. Chờ ~30–50 phút. Log cuối phải in:
   `DONE games_with_level=15/25` (nếu thấy số thấp hơn nhiều → báo lại, có thể path/data sai).
7. Sau khi commit xong (notebook "Complete") → tab **Submit** → chọn version vừa chạy → Submit.
   - Hoặc CLI: `kaggle competitions submit -c arc-prize-2026-arc-agi-3 -k thnhnguyn2001/<NOTEBOOK> -v <VERSION> -m "explore2 15/25"`
8. **GHI LẠI submission ID** hiện trên trang My Submissions (cần cho Paper Track).
9. **Make notebook PUBLIC** (Share → Public) — bắt buộc để xét giải, và phải public **trước 30/06** cho Milestone 1.
   - Tối đa **5 submission/ngày**, chọn **2 Final Submission**.

## PHẦN B — Nộp Paper Track (lấy giải paper, có cửa cao hơn)

1. **Vào** kaggle.com/competitions/arc-prize-2026-paper-track → **Join**.
2. **New Writeup**.
3. **Tiêu đề/nội dung:** dán từ `D:\arc_agi_3\PAPER_TRACK_WRITEUP.md`
   (tiêu đề = dòng `#` đầu; subtitle = dòng `**Subtitle:**`; body = phần còn lại).
4. **Media Gallery → Cover image (BẮT BUỘC):** upload `D:\arc_agi_3\cover.png`.
5. **Attach Public Notebook:** link tới notebook ở Phần A (đã public).
6. **Điền 2 chỗ trống** trong writeup: **submission ID** (bước A8) + **link notebook**.
7. Chọn **Track** (chỉ có 1 track) → **Submit** (nút góc phải trên).
   - Hackathon: mỗi team **1 submission**.

## LƯU Ý LICENSE & RULE
- **Winner license = CC-BY 4.0** (OSI-approved). Code mình MIT → tương thích; trong notebook/writeup ghi attribution: dựa trên ARC-AGI-3-Agents (MIT, ARC Prize) + kỹ thuật mask/reorder port từ Occam (g-baskin/occam, MIT, Sean Donahoe).
- **1 tài khoản Kaggle duy nhất** — không nộp từ nhiều account (DQ).
- Deadline: ARC-AGI-3 entry **26/10**, final **02/11**; Paper Track final **09/11**.
- Milestone notebook phải PUBLIC trước **30/06** & **30/09**.

## NẾU NOTEBOOK LỖI KHI CHẠY TRÊN KAGGLE
- "wheels not found": path mount khác → sửa `DATA_ROOT` ở đầu cell cho khớp panel Data.
- Import lỗi: xác nhận đã Add Input đúng competition dataset (có cả `arc_agi_3_wheels/` và `environment_files/`).
- Hết giờ (>9h): hạ `MAX_STEPS` (đầu phần 3) từ 15000 xuống 8000 — vẫn ~14-15/25.
- Gửi lại em log lỗi, em sửa cell.
