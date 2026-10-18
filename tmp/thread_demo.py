import threading 
import time 

def chao(ten: str, delay: float) -> None:
    time.sleep(delay)
    print(f" [thread] Xin chào {ten} (sau {delay} giây)")

print("1) Chạy 2 việc song song thay vì tuần tự")
start = time.perf_counter()
t1 = threading.Thread(target=chao, args=("Alice", 1))
t2 = threading.Thread(target=chao, args=("Bob", 1))
t1.start()
t2.start()
t1.join()
t2.join()
print(f" -> tong thoi gian: {time.perf_counter() - start:.2f}s")

#------------------------------------------------------------------------
# 2) Lock
counter_khong_lock = 0
counter_co_lock = 0
lock = threading.Lock()

def tang_khong_lock() -> None:
    global counter_khong_lock
    for _ in range(200000):
        counter_khong_lock += 1 # doc-sua-ghi, co the bi xem ke

def tang_co_lock() -> None:
    global counter_co_lock
    for _ in range(200000):
        with lock:
            counter_co_lock += 1 # doc-sua-ghi, co the bi xem ke

print("2) Lock bao ve du lieu dung chung")
threads = [threading.Thread(target=tang_khong_lock) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"  không lock: {counter_khong_lock}  (kỳ vọng 800000, thường bị thiếu)")

threads = [threading.Thread(target=tang_co_lock) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"  có lock: {counter_co_lock}  (kỳ vọng 800000, thường không bị thiếu)")

# ------------------------------------------------- 3) Mô phỏng job nền (giống jobs.py)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

def submit(name, fn, *args) -> str:
    job_id = name.lower()
    with _lock:
        _jobs[job_id] = {"status": "queued"}

    def run() -> None:
        with _lock:
            _jobs[job_id].update(status="done", result=result)
        result = fn(*args)
        with _lock:
            _jobs[job_id].update(status="done", result=result)

    threading.Thread(target=run, daemon=True).start()
    return job_id 

def viec_dai(x: int) -> str:
    time.sleep(2)
    return x * x 
print("3) Job nền: trả về ngay, không chặn, rồi poll trạng thái")
jid = submit("square", viec_dai, 7)
print(f"  submit trả ngay: {jid}")

for _ in range(4):
    time.sleep(0.4)
    with _lock:
        print(f"  poll: {dict(_jobs[jid])}")

    