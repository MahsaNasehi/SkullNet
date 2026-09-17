# آموزش شکستگی جمجمه با YOLO26s-P2

این pipeline فقط برش‌هایی را وارد loss یولو می‌کند که فایل JSON annotation دارند. در نتیجه Studyهای بدون annotation مکانی به‌اشتباه negative محسوب نمی‌شوند. تقسیم `train/val/test` در سطح `PatientID` و به‌صورت stratified انجام می‌شود؛ تمام CTهای یک بیمار فقط در یک split قرار می‌گیرند.

آمار واقعی نسخه‌ی حاضر داده ۱۹۸ Study برچسب‌دار متعلق به ۱۸۳ بیمار است (۵٬۱۷۶ برش annotation‌شده، شامل ۲۶۰ برش مثبت و ۳۵۶ box). ۱۴۰ Study بدون annotation از آموزش detector حذف شده‌اند.

ورودی از DICOM به HU تبدیل می‌شود و window ثابت `WL=800, WW=1600` دارد، یعنی بازه‌ی `[0, 1600] HU`. ورودی پیش‌فرض 2.5D است: برش مرکزی و نزدیک‌ترین همسایه‌ها در فاصله‌ی فیزیکی ۵ میلی‌متر سه کانال تصویر را می‌سازند.

نسخه‌ی رسمی P2 وزن آماده ندارد. اسکریپت مدل رسمی `YOLO26s-P2` را می‌سازد و پارامترهای هم‌شکل `yolo26s.pt` آموزش‌دیده روی COCO را به آن منتقل می‌کند. نتیجه‌ی انتقال در `outputs/<run>/pretrained_transfer_audit.json` ذخیره می‌شود.

## آماده‌سازی محیط و داده

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# نمونه برای GPU و درایور CUDA 12.4 این سرور:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python scripts/download_weights.py
PYTHONPATH=src python src/prepare_yolo26_dataset.py
```

برای rebuild دیتاست موجود، فقط در صورت اطمینان `--force` اضافه شود. خلاصه‌ی split و کنترل leakage در `data_prepared/skull_hu800_ww1600/split_summary.json` ثبت می‌شود.

## آموزش

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
source .venv/bin/activate
PYTHONPATH=src python src/train_yolo26s_p2.py \
  --device 0 --batch 16 --imgsz 768 --epochs 150 --workers 0
```

در این سرور `/dev/shm` فقط حدود ۴۴۲ مگابایت است؛ بنابراین `workers=0` ضروری است و از خطای `Bus error` جلوگیری می‌کند. در صورت کمبود VRAM نیز `--batch 8` بدهید. برای ادامه‌ی اجرای قطع‌شده:

```bash
PYTHONPATH=src python src/train_yolo26s_p2.py \
  --resume outputs/yolo26s_p2_hu800_ww1600/weights/last.pt
```

test تا پایان انتخاب مدل دست‌نخورده بماند. پس از پایان انتخاب مدل، فقط یک بار:

```bash
PYTHONPATH=src python src/evaluate_yolo26_test.py \
  --weights outputs/yolo26s_p2_hu800_ww1600/weights/best.pt \
  --device 0 --batch 16 --imgsz 768
```

## آموزش پیشنهادی پنج‌فولدی

Test ثابت قبلی دست‌نخورده می‌ماند. ۱۵۵ بیمار train+validation در پنج fold کاملاً مجزای بیمار تقسیم می‌شوند. الگوریتم علاوه بر تعداد بیمار مثبت، تعداد برش مثبت و boxها را نیز متوازن می‌کند.

```bash
PYTHONPATH=src python src/prepare_yolo26_kfold.py

PYTHONPATH=src nohup python src/train_yolo26s_p2_kfold.py \
  --fold all --device 0 --batch 16 --imgsz 768 \
  --epochs 150 --workers 0 --parallel 2 \
  > outputs/train_yolo26s_p2_kfold.log 2>&1 &
```

پنج مدل در پوشه‌های `outputs/yolo26s_p2_hu800_ww1600_fold0` تا `fold4` ذخیره می‌شوند. با `--parallel 2` ابتدا foldهای ۰ و ۱، سپس ۲ و ۳، و در پایان fold ۴ اجرا می‌شوند. لاگ مستقل هر مدل در `outputs/kfold_logs/fold_N.log` است. برای ادامه‌ی foldهایی که `last.pt` دارند، گزینه‌ی `--resume-existing` را اضافه کنید.

## آزمایش نمونه‌گیری متوازن Fold 0

جزئیات ایده، محدودیت‌ها، تغییرات، تست‌ها و نتیجه واقعی اجرا در [agent.md](agent.md) ثبت می‌شود. این آزمایش همان مدل COCO و تنظیمات HU را با ۲۵٪ تصاویر اصلی مثبت، توازن سهم بیماران و چرخش منفی‌ها آموزش می‌دهد. نسبت پیش از mosaic است. تعداد تصاویر هر epoch و تعداد batchها همان baseline باقی می‌ماند. validation کامل و بدون نمونه‌گیری است.

برای شروع **فقط اگر قبلاً اجرا نشده است**، روی ترمینال دارای دسترسی GPU:

```bash
PYTHONPATH=src setsid nohup .venv/bin/python -u scripts/run_balanced_fold0.py \
  --device 0 >> outputs/balanced_fold0.log 2>&1 < /dev/null &
```

```bash
tail -f outputs/balanced_fold0.log
```

پس از پایان آموزش، runner خودکار `best.pt` را روی validation فولد صفر ارزیابی و `comparison_to_baseline.json` را در `outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/` ذخیره می‌کند. شروع و پایان مراحل و نتایج نیز به `agent.md` اضافه می‌شوند. test نهایی اجرا نمی‌شود.

برای ادامه اجرای قطع‌شده همان runner را با `--resume` و برای تکرار صرفاً ارزیابی با `--evaluate-only` اجرا کنید. checkpointهای تکمیل‌شده و فاقد optimizer قابل resume نیستند. قفل آزمایش از اجرای هم‌زمان تکراری جلوگیری می‌کند.

بررسی سیاست نمونه‌گیری بدون آموزش و بدون نیاز به GPU:

```bash
PYTHONPATH=src .venv/bin/python src/train_yolo26s_p2.py \
  --data data_prepared/skull_hu800_ww1600/kfold/fold_0.yaml \
  --positive-fraction 0.25 --batch 16 --workers 0 --sampling-audit-only
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python scripts/smoke_balanced_training.py
```

تست آخر روی زیرمجموعه کوچک train/validation در CPU، دو epoch در 128px، ذخیره checkpoint، قطع عمدی و resume را بررسی می‌کند و ارزیابی دقت مدل نیست.

## QWK مستقل شکستگی روی test ثابت

این ارزیابی یک بار پس از انتخاب پروتکل انجام شده است. معیار، Binary QWK وجود شکستگی در سطح Study است و ICH/MLS را نمی‌خواند. پروتکل اصلی از میانگین confidence پنج مدل K-fold روی هر slice، سپس `top3_mean` در Study و threshold ثابت 0.5 استفاده می‌کند. خروجی در `outputs/fixed_test_fracture_kfold_ensemble/` ذخیره شده است.

نمایش نتیجه موجود، بدون inference دوباره:

```bash
.venv/bin/python -m json.tool outputs/fixed_test_fracture_kfold_ensemble/metrics.json
column -s, -t < outputs/fixed_test_fracture_kfold_ensemble/study_predictions.csv | less -S
```

فرمان بازتولید در محیط/پوشه خروجی تازه:

```bash
PYTHONPATH=src .venv/bin/python src/evaluate_fracture_test_qwk.py \
  --device 0 --batch 16 --imgsz 768
```

اسکریپت اگر `metrics.json` قبلی موجود باشد عمداً از overwrite و اجرای تکراری test جلوگیری می‌کند. خروجی تک‌مدل‌ها فقط diagnostic است و برای انتخاب fold یا threshold پس از دیدن test استفاده نمی‌شود.
## انتخاب threshold شکستگی فقط با OOF

پس از پایان هر پنج Fold و تولید `study_evaluation/study_predictions.csv`، آستانه‌ی Binary QWK را بدون sweep روی test انتخاب کنید:

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
source .venv/bin/activate
PYTHONPATH=src python src/select_oof_fracture_threshold.py
python -m json.tool outputs/oof_fracture_threshold_v1/threshold_report.json
column -s, -t < outputs/oof_fracture_threshold_v1/fixed_test_secondary_predictions.csv | less -S
```

اسکریپت علاوه بر threshold نهایی، ارزیابی cross-fitted (انتخاب threshold هر Fold فقط از چهار Fold دیگر)، PR-AUC، ROC-AUC، confusion matrix عددی و bootstrap بیمارمحور را ذخیره می‌کند. اجرای مجدد روی مسیر خروجی موجود عمداً متوقف می‌شود؛ برای آزمایش development جدید از `--output` با نام جدید استفاده کنید. تحلیل test این پوشه ثانویه/post-hoc است، چون نتیجه‌ی test با threshold=0.5 قبلاً مشاهده شده بود.

Audit تشخیصی سه FN و یک FP فعلی:

```bash
PYTHONPATH=src python src/audit_fracture_test_errors.py
python -m json.tool outputs/fixed_test_error_audit_v1/audit_report.json
column -s, -t < outputs/fixed_test_error_audit_v1/study_model_summary.csv | less -S
```

خروجی‌های عددی، overlayها و تفسیر کامل در `outputs/fixed_test_error_audit_v1/AUDIT_SUMMARY_FA.md` قرار دارند. این audit post-hoc است و نباید از چهار test case برای tune کردن مدل استفاده شود.

ارزیابی baseline proposal recall روی پنج OOF Fold:

```bash
PYTHONPATH=src python src/evaluate_oof_proposal_recall.py
python -m json.tool outputs/oof_proposal_recall_run_a_768/metrics.json
column -s, -t < outputs/oof_proposal_recall_run_a_768/runtime_by_fold.csv
```

خلاصه Run A در `outputs/oof_proposal_recall_run_a_768/BASELINE_SUMMARY_FA.md` و مقادیر خام هر GT در `gt_proposal_matches.csv` ذخیره شده‌اند.
