# ARC Prize 2026 — ARC-AGI-3 (repo riêng)

Mục tiêu: lọt **top score** mốc Milestone (30/06/2026 & 30/09/2026), không nhắm Grand Prize (cần 100%, bất khả thi).
Kiến trúc đã chọn: **Graph-based exploration** (KHÔNG LLM, KHÔNG cần train GPU). Dev trên laptop → long-run trên bee.

## Sự thật về cuộc thi
- Game lưới **64×64, 16 màu**. Actions: RESET, ACTION1-4 (di chuyển), ACTION5 (tương tác), ACTION6 (click toạ độ x,y 0-63), ACTION7 (undo, có thể không bật).
- Mỗi frame: game_id, frame (3D grid), state (NOT_PLAYED/NOT_FINISHED/WIN/GAME_OVER), score, win_score, available_actions.
- Không có hướng dẫn/luật — agent tự khám phá cơ chế, tự tìm "thắng là gì", mang kinh nghiệm sang level khó hơn.
- Frontier AI hiện tại ~0.51%, người chơi 100%. Preview top-1 RL ~12.6%, top-3 graph-explore median 17/25 levels.
- Runtime budget tới $10K (được gọi API thương mại). Phải open-source lời giải.
- Nộp qua Kaggle notebook; thường gọi compute ngoài (Modal/Lambda/RunPod). Final deadline 02/11/2026, kết quả 04/12/2026.

## Thuật toán (theo "just explore" top-3, sẽ cải tiến)
1. RESET vào level.
2. Graph trạng thái: node = hash(frame), edge = action -> frame mới. Mỗi node lưu tập action đã thử.
3. Frame processor: segment lưới theo connected-component cùng màu, che status bar, gom phần tử tương tác thành ~5 nhóm ưu tiên (khả năng là "nút bấm") -> sinh ứng viên (x,y) cho ACTION6.
4. Chọn action: ưu tiên action chưa thử của node hiện tại (simple trước, rồi ACTION6 theo nhóm ưu tiên). Nếu node cạn action -> BFS sang frontier gần nhất, replay path, tiếp tục.
5. Phát hiện tiến triển: score/level tăng -> giữ nhánh đó. Carry graph/heuristic sang level sau.

## Hạn chế đã biết của baseline (chỗ ta sẽ thắng)
- Kém với game phi tất định (non-deterministic).
- State space lớn -> cần world-model/abstraction.
- Status bar layout lạ làm hỏng segmentation.
Hướng cải tiến: (a) world-model nhẹ dự đoán frame kế; (b) abstraction lưới (objects thay vì pixel); (c) intrinsic-motivation có kiểm soát.

## Trạng thái hạ tầng
- Repo riêng: D:\arc_agi_3 (git init riêng, remote `upstream` = repo gốc arcprize để pull update framework).
- venv: .venv (Python 3.13.5). Deps cài qua `pip install -e .`.
- .env đã tạo từ .env.example — CẦN điền ARC_API_KEY.

## Việc CHỈ BẠN làm được
1. Vào https://three.arcprize.org/ đăng ký lấy **ARC_API_KEY**, dán vào D:\arc_agi_3\.env (dòng ARC_API_KEY=...).
2. (sau) Tạo tài khoản Kaggle, join 2 competition để nộp.

## Lệnh chạy
- Baseline random: `.venv\Scripts\python main.py --agent=random --game=ls20`
- (Agent của ta sẽ là `--agent=explore` sau khi build xong.)

## Lưu ý môi trường (session này)
- Tool Write/Read và `cat` file repo lớn KHÔNG đáng tin (overlay sandbox lệch disk thật).
- Kênh tin cậy: Bash chế độ ra-disk-thật để GHI + CHẠY; introspect interface bằng `python -c "import inspect..."`.
