# ارزیابی فقط مدل Fracture

## آیا قواعد مستقل شدنی است؟

بله. از مجموعهٔ قواعد رسمی، هر شاخه‌ای که به ICH یا MLS نیاز دارد حذف می‌شود؛
تنها تصمیمِ باقی‌ماندهٔ شکستگی به‌صورت قواعد **standalone** نوشته می‌شود:

```python
# standalone_fracture_rules.py  — بدون ICH، بدون MLS، بدون فراخوانی تابع رسمی
if fracture_prob >= 0.5:
    return 1  # Urgent
return 0      # Non-urgent
```

این قواعد ورودی دیگری ندارند؛ فقط دقت مدل شکستگی شما را می‌سنجند.

## فایل‌های اصلی

| فایل | نقش |
|---|---|
| `standalone_fracture_rules.py` | قواعد مستقل (پاسخ خواستهٔ شما) |
| `evaluate_standalone_fracture.py` | ارزیابی فقط مدل روی OOF |
| `results_standalone/` | خروجی accuracy / Macro-F1 / sens-spec |
| `evaluate_fracture_only.py` | نسخهٔ قبلی: تابع رسمی + صفر کردن ICH/MLS |

## اجرا (پیشنهادی)

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
source .venv/bin/activate
python eval_fracture_only/evaluate_standalone_fracture.py --overwrite
```

خروجی: `eval_fracture_only/results_standalone/SUMMARY.md`
