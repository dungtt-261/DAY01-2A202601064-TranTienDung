"""
K4 — Ngày 1: Khám Phá LLM API (14h00–18h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
}

# Luồng chính: OpenAI (mặc định, không cần đặt gì trong .env).
# Không có key OpenAI? Dùng luồng thay thế Google Gemini (Phụ lục B
# trong LAB_GUIDE.md) — tên model đổi qua .env. NVIDIA NIM: Phụ lục C.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 15h00–15h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model OpenAI sử dụng (mặc định: gpt-4o).
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        from openai import OpenAI            # import BÊN TRONG hàm
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # đo thời gian bằng time.time() trước và sau lời gọi API
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    request_messages = [{"role": "user", "content": prompt}]

    started_at = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=request_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    elapsed_seconds = max(time.time() - started_at, 1e-12)
    answer = completion.choices[0].message.content

    return answer, float(elapsed_seconds)


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        Tái sử dụng call_openai() với model=OPENAI_MINI_MODEL — 1 dòng code.
    """
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gpt4o_answer":      str
            - "mini_answer":       str
            - "gpt4o_time":       float
            - "mini_time":        float
            - "gpt4o_cost": float  (USD ước tính cho phản hồi)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(
            OPENAI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"]
        )
        cost = (len(response.split()) / 0.75) / 1000 * pricing["output"]
        (0.75 từ ≈ 1 token — ước lượng thô; Part 2 sẽ tính chính xác hơn.
         Dùng .get để lấy đúng giá model đang chạy — gpt-4o, gemini...;
         model không có trong bảng thì lấy giá gpt-4o làm tham chiếu)
    """
    large_answer, large_latency = call_openai(prompt)
    small_answer, small_latency = call_openai_mini(prompt)

    fallback_price = PRICING_PER_1K_TOKENS["gpt-4o"]
    active_price = PRICING_PER_1K_TOKENS.get(OPENAI_MODEL, fallback_price)
    estimated_output_tokens = len(large_answer.split()) / 0.75
    estimated_cost = (
        estimated_output_tokens * active_price["output"] / 1000
    )

    comparison = {
        "gpt4o_answer": large_answer,
        "mini_answer": small_answer,
        "gpt4o_time": large_latency,
        "mini_time": small_latency,
        "gpt4o_cost": estimated_cost,
    }
    return comparison


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 15h40–16h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm 2 phần: system prompt (định hình vai trò/persona
    của model) và user prompt (câu hỏi thật).

    Args:
        system_prompt: Chỉ dẫn vai trò, ví dụ "Bạn là giáo viên tiểu học,
                       giải thích mọi thứ thật đơn giản."
        user_prompt:   Tin nhắn của người dùng.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    started_at = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    duration = max(time.time() - started_at, 1e-12)
    content = completion.choices[0].message.content

    return content, float(duration)


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).

    Gợi ý:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

        tiktoken cần tải bộ mã hóa từ mạng ở lần chạy đầu. Hãy bọc trong
        try/except — nếu lỗi (offline, model lạ), dùng ước lượng dự phòng:
        max(1, len(text) // 4)   (trung bình 1 token ≈ 4 ký tự)
    """
    fallback_count = max(1, len(text) // 4)

    try:
        import tiktoken

        tokenizer = tiktoken.encoding_for_model(model)
        encoded_text = tokenizer.encode(text)
    except Exception:
        return fallback_count

    return len(encoded_text)


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS — tách riêng chi phí
    input (prompt) và output (response).

    Returns:
        Dict với các key:
            - "prompt_tokens":  int
            - "completion_tokens": int
            - "prompt_cost":    float  (USD)
            - "completion_cost":   float  (USD)
            - "total_cost":    float  (USD)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
        prompt_cost = prompt_tokens / 1000 * pricing["input"]
        (.get với fallback: model không có trong bảng giá — ví dụ model NIM
         miễn phí — thì lấy giá gpt-4o làm tham chiếu học tập)
    """
    token_usage = {
        "prompt_tokens": count_tokens(prompt, model),
        "completion_tokens": count_tokens(response, model),
    }
    default_pricing = PRICING_PER_1K_TOKENS["gpt-4o"]
    model_pricing = PRICING_PER_1K_TOKENS.get(model, default_pricing)

    input_cost = (
        token_usage["prompt_tokens"] / 1000 * model_pricing["input"]
    )
    output_cost = (
        token_usage["completion_tokens"] / 1000 * model_pricing["output"]
    )

    return {
        **token_usage,
        "prompt_cost": input_cost,
        "completion_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 16h30–17h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.

    Hành vi:
        - Stream token từ OpenAI ngay khi chúng được sinh ra (in từng chunk).
        - Duy trì 4 lượt hội thoại gần nhất trong history.
        - Gõ 'quit', 'exit' hoặc 'bye' để thoát.

    Gợi ý:
        - Giữ list `history` gồm các dict {"role": ..., "content": ...}.
        - Dùng stream=True trong client.chat.completions.create() và lặp:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - Sau mỗi lượt, thêm phản hồi assistant vào history.
        - Cắt history còn 4 lượt cuối (8 message): history = history[-8:]
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict] = []
    exit_commands = {"quit", "exit", "bye"}

    while True:
        user_message = input("Bạn: ")
        if user_message.strip().lower() in exit_commands:
            return

        request_messages = [
            *history,
            {"role": "user", "content": user_message},
        ]
        response_stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=request_messages,
            stream=True,
        )

        print("Trợ lý: ", end="", flush=True)
        reply_parts = []
        for chunk in response_stream:
            text_part = chunk.choices[0].delta.content or ""
            reply_parts.append(text_part)
            print(text_part, end="", flush=True)
        print()

        assistant_message = "".join(reply_parts)
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        del history[:-8]


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với
    exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Callable không tham số.
        max_retries: Số lần thử lại tối đa.
        base_delay:  Delay ban đầu (giây) trước lần thử lại đầu tiên.

    Returns:
        Giá trị trả về của fn() khi thành công.

    Raises:
        Exception cuối cùng của fn() sau khi hết số lần thử.
    """
    attempt = 0

    while True:
        try:
            return fn()
        except Exception:
            if attempt >= max_retries:
                raise

            wait_seconds = base_delay * (2 ** attempt)
            time.sleep(wait_seconds)
            attempt += 1


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 17h10–17h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.

    Hành vi:
        1. Dùng `persona` làm system prompt cho TOÀN BỘ phiên chat.
        2. Mỗi lượt: đọc tin nhắn qua get_input(); nếu là 'quit'/'exit'/'bye'
           (không phân biệt hoa thường) → kết thúc phiên.
        3. Gọi API với stream=True, messages = system + history + tin nhắn mới.
           Bọc lời gọi API trong retry_with_backoff để chịu lỗi tạm thời.
        4. In từng chunk khi stream về, ghép lại thành reply hoàn chỉnh.
        5. Cập nhật history (user + assistant), giữ tối đa 4 lượt cuối
           (8 message): history = history[-8:]
        6. Cộng dồn thống kê bằng count_tokens và estimate_cost.
        7. Dừng khi đạt max_turns (nếu được đặt).

    Args:
        persona:   Mô tả vai trò, dùng làm system prompt.
        get_input: Hàm đọc input (mặc định: input). Tham số này giúp
                   test tự động không cần bàn phím thật.
        max_turns: Số lượt tối đa (None = không giới hạn).

    Returns:
        Dict thống kê phiên chat:
            - "turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "tokens_used": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 8 message)

    Gợi ý khung sườn:
        if get_input is None:
            get_input = input
        history, turns, tokens_used, total_cost = [], 0, 0, 0.0
        while True:
            if max_turns is not None and turns >= max_turns:
                break
            user_msg = get_input()
            if user_msg.strip().lower() in ("quit", "exit", "bye"):
                break
            messages = [{"role": "system", "content": persona}] + history \\
                       + [{"role": "user", "content": user_msg}]
            # stream = retry_with_backoff(lambda: client.chat...create(
            #              model=..., messages=messages, stream=True))
            # reply = ghép các chunk...
            ...
        return {"turns": turns, "tokens_used": tokens_used,
                "total_cost": total_cost, "history": history}
    """
    input_reader = get_input or input

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stop_words = {"quit", "exit", "bye"}
    history: list[dict] = []
    turns = 0
    tokens_used = 0
    total_cost = 0.0

    while max_turns is None or turns < max_turns:
        user_message = input_reader()
        if user_message.strip().lower() in stop_words:
            break

        api_messages = [
            {"role": "system", "content": persona},
            *history,
            {"role": "user", "content": user_message},
        ]

        def create_stream():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=api_messages,
                stream=True,
            )

        chunks = retry_with_backoff(create_stream)

        print("Trợ lý: ", end="", flush=True)
        reply_parts = []
        for chunk in chunks:
            piece = chunk.choices[0].delta.content or ""
            reply_parts.append(piece)
            print(piece, end="", flush=True)
        print()

        assistant_reply = "".join(reply_parts)
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_reply},
            ]
        )
        history = history[-8:]

        turn_cost = estimate_cost(
            user_message,
            assistant_reply,
            model=OPENAI_MODEL,
        )
        tokens_used += (
            turn_cost["prompt_tokens"] + turn_cost["completion_tokens"]
        )
        total_cost += turn_cost["total_cost"]
        turns += 1

    return {
        "turns": turns,
        "tokens_used": tokens_used,
        "total_cost": total_cost,
        "history": history,
    }


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    comparisons = []

    for original_prompt in prompts:
        row = compare_models(original_prompt)
        row_with_prompt = dict(row)
        row_with_prompt["prompt"] = original_prompt
        comparisons.append(row_with_prompt)

    return comparisons


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    Gợi ý: cắt text dài còn 40 ký tự cho dễ nhìn.
    """
    def clip(value: object, width: int = 40) -> str:
        one_line = " ".join(str(value).splitlines())
        if len(one_line) <= width:
            return one_line
        return f"{one_line[: width - 3]}..."

    columns = (
        ("Prompt", "prompt"),
        ("GPT-4o Response", "gpt4o_answer"),
        ("Mini Response", "mini_answer"),
        ("GPT-4o Latency", "gpt4o_time"),
        ("Mini Latency", "mini_time"),
    )

    formatted_rows = []
    for item in results:
        row = []
        for _, key in columns:
            value = item.get(key, "")
            if key.endswith("_time"):
                value = f"{float(value):.3f}s"
            row.append(clip(value))
        formatted_rows.append(row)

    headers = [title for title, _ in columns]
    column_widths = [
        max(
            len(headers[index]),
            *(len(row[index]) for row in formatted_rows),
        )
        if formatted_rows
        else len(headers[index])
        for index in range(len(headers))
    ]

    def draw_row(values: list[str]) -> str:
        return " | ".join(
            value.ljust(column_widths[index])
            for index, value in enumerate(values)
        )

    divider = "-+-".join("-" * width for width in column_widths)
    output_lines = [draw_row(headers), divider]
    output_lines.extend(draw_row(row) for row in formatted_rows)
    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
