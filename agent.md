# گزارش پیاده‌سازی، آزمایش‌ها و تصمیم‌های پروژه

این فایل طبق درخواست کاربر، دفتر ثبت دقیق کار است. هر تغییر جدید باید همراه با ایده، دلیل، فایل‌های درگیر، روش بررسی، نتیجه واقعی، محدودیت‌ها و قدم بعدی به این گزارش اضافه شود. نتیجه یک تست نرم‌افزاری با افزایش دقت مدل یکسان نیست؛ اجرای برنامه نیز با تکمیل آموزش یکسان نیست.

## آزمایش B25 — نمونه‌گیری متوازن بیمارمحور در Fold 0

شروع ثبت: 2026-09-11، ساعت حدود 23:59 به وقت تهران.

درخواست: «بهترین ایده را الان برای تست انجام بده». تصمیم: یک آزمایش محدود روی Fold 0 با همان YOLO26s-P2 و COCO initialization، همراه با افزایش سهم تصاویر مثبت. کاربر سپس خواست همه پیاده‌سازی‌ها، تست‌ها، ایده‌ها و دلیل تصمیم‌ها در همین فایل ثبت شوند. این درخواست به کار جاری اضافه شده است.

### شواهد و دلیل انتخاب

- اطلاعات اجرای فعلی با گزارش قدیمی `information /FRACTURE_QWK_99_REVIEW_FA.md` و ممیزی پروژه YOLO11 متفاوت است. مدل این اجرا YOLO26s-P2 است و تقسیم آن یک test ثابت به‌همراه پنج fold بیمارمحور است.
- Fold 0 در train دارای ۳۵۱۸ تصویر یکتا است: ۱۷۷ تصویر مثبت و ۳۳۴۱ تصویر منفی دارای annotation. مثبت‌ها متعلق به ۱۹ بیمارند؛ مجموع بیماران train برابر ۱۲۴ است.
- سهم فعلی تصاویر مثبت حدود ۵٫۰۳٪ است. فرضیه آزمایش این است که افزایش سهم تصاویر مثبت، فرصت یادگیری شکستگی را بیشتر می‌کند و می‌تواند حساسیت مطالعه را بهبود دهد. این فرضیه هنوز با نتیجه آموزش جدید تأیید نشده است.
- validation فولد صفر ۹۰۰ تصویر، ۳۱ بیمار و ۳۳ مطالعه دارد. پنج مطالعه مثبت است. baseline در threshold=0.5 و تجمیع top3_mean دارای TP=2، FP=0، FN=3، TN=28، AP مطالعه=0.5366507 و QWK ایزوله=0.9751693 است.
- QWK baseline همیشه‌منفی همین fold نیز 0.9751693 است. افزایش QWK به‌تنهایی برای اثبات تشخیص بهتر شکستگی کافی نیست.
- در بررسی قبلی همین گفتگو، جایگزینی max به‌جای top3، AP را به 0.5487719 رساند ولی TP/FP/FN تغییر نکرد. بیشینه confidence سه مطالعه ازدست‌رفته حدود 0.03519، 0.00333 و 0.00991 بود. بنابراین تغییر تجمیع به‌تنهایی شواهد قوی بازیابی FNها نداشت.
- Verifier، تغییر رزولوشن، SAHI و بزرگ‌کردن مدل در این آزمایش فعال نمی‌شوند؛ این آزمایش مشخصاً سیاست نمونه‌گیری را می‌سنجد. این موارد ایده‌های مراحل بعدی‌اند، نه پیاده‌سازی انجام‌شده.

### طراحی آزمایش

| مؤلفه | تنظیم |
|---|---|
| مدل | YOLO26s-P2؛ همان معماری و headهای P2/P3/P4/P5 |
| شروع آموزش | وزن COCO محلی `weights/yolo26s.pt`؛ بدون شروع از best.pt آموزش‌دیده Fold 0 |
| ورودی | HU در [0,1600]، WL=800، WW=1600، همان 2.5D موجود |
| اندازه و batch | imgsz=768، batch=16 |
| optimizer و augmentation | همان تنظیمات `src/train_yolo26s_p2.py`؛ mosaic=0.30 نیز حفظ شده |
| بودجه | حداکثر 150 epoch، patience=30، تعداد تصاویر ورودی هر epoch برابر 3518 و تعداد batch برابر 220 |
| نسبت جدید | ۴ تصویر مثبت و ۱۲ تصویر منفی در batch کامل؛ در batch آخر ۱۴تایی، ۴ مثبت و ۱۰ منفی |
| مجموع epoch | ۸۸۰ برداشت مثبت و ۲۶۳۸ برداشت منفی؛ نسبت واقعی 0.250142 |
| گروه‌بندی | توازن تعداد برداشت از بیماران داخل هر کلاس؛ چرخش در برش‌های هر بیمار |
| validation/test | validation کامل همان Fold 0؛ بدون نمونه‌گیری متوازن برای validation و بدون inference روی test |
| نام اجرا | `yolo26s_p2_hu800_ww1600_fold0_balanced25` |

نسبت ۲۵٪ مربوط به تصاویر اصلی انتخاب‌شده **پیش از mosaic و augmentation** است. mosaic می‌تواند تصاویر کمکی اضافه کند و هندسه ممکن است باکس‌ها را برش دهد؛ در نتیجه نسبت نهایی تصاویر دارای باکس یا تعداد instanceهای چاپ‌شده در هر batch الزاماً دقیقاً ۲۵٪ نیست. هیچ class-loss weight اضافی اعمال نشده است.

تعداد batch و سقف epoch مشابه baseline است؛ early stopping ممکن است در زمان متفاوت رخ دهد، بنابراین مجموع گام‌های نهایی الزاماً برابر نیست. برای نتیجه‌گیری ظرفیت/دقت، زمان و epoch واقعی هر دو اجرا نیز باید گزارش شود.

### پیاده‌سازی انجام‌شده

1. **`src/patient_balanced_sampling.py` — sampler و اعتبارسنجی داده**
   - خواندن فهرست‌های صریح train/val/test و تطبیق با manifest؛ رد تصاویر تکراری، بیمار نامشخص، هم‌پوشانی بیمار و ورود test ثابت به development.
   - بررسی وجود تصویر و JSON annotation برای هر تصویر train و برابری تعداد labelها با manifest؛ تصاویر بدون annotation به pool منفی اضافه نمی‌شوند.
   - ثبت SHA256 مربوط به YAML، manifest، فهرست splitها و labels آموزش برای تشخیص تغییر داده در ادامه آموزش.
   - ترتیب deterministic مستقل از RNG آموزش: چرخش بین بیماران و سپس بین برش‌های هر بیمار؛ shuffle داخل هر batch.
   - بازسازی ترتیب epoch با seed و شماره epoch؛ برای resume نیازی به تاریخچه generator نیست.
   - ثبت تعداد برداشت، نمونه یکتا، بیماران دیده‌شده، سقف سهم هر بیمار و hash ترتیب هر epoch.

2. **`src/balanced_yolo_trainer.py` — اتصال به Ultralytics نصب‌شده**
   - subclass از DetectionTrainer؛ فقط loader آموزش جایگزین می‌شود.
   - loader محدود به هر epoch برای جلوگیری از ورود batchهای پیش‌خوانده‌شده epoch بعد به سیاست جدید؛ workers=0 و یک دستگاه.
   - hook آغاز epoch، شماره epoch sampler را تنظیم و `sampling_epochs.jsonl` را ثبت می‌کند.
   - تطبیق ترتیب واقعی dataset با manifest و تعداد باکس‌ها، جلوگیری از حذف خاموش نمونه‌ها.
   - ذخیره policy در `sampling_policy.json` و خود مدل برای حفظ آن در checkpoint/EMA و resume.
   - کاهش خودکار batch یا DDP برای این آزمایش پشتیبانی نمی‌شود، چون نسبت و بودجه آزمایش را تغییر می‌دهد.

3. **`src/train_yolo26s_p2.py` — گزینه‌های opt-in آزمایش**
   - اضافه‌شدن `--positive-fraction`، `--sampling-manifest` و `--sampling-audit-only`.
   - بدون `--positive-fraction`، آموزش جدید همان loader عادی قبلی را دارد.
   - resume در صورت وجود policy، trainer متوازن را خودکار فعال می‌کند. تغییر نسبت هنگام resume رد می‌شود.
   - checkpoint تکمیل‌شده و فاقد optimizer به‌عنوان resume رد می‌شود تا به‌اشتباه آموزش تازه شروع نشود.
   - initialization این آزمایش پیش از ساخت مدل seed می‌شود؛ وزن اولیه P2 در پوشه اجرای جدید ذخیره می‌شود و فایل مشترک initialization قبلی بازنویسی نمی‌شود.

4. **`src/evaluate_fold_qwk.py` — اصلاح یک ناسازگاری واقعی در threshold**
   - قبلاً `--threshold` روی معیار دودویی اعمال می‌شد ولی QWK اصلی همواره raw score را با 0.5 می‌سنجید.
   - اکنون همان threshold به تصمیم fracture تبدیل می‌شود و سپس تابع رسمی مسابقه، با آستانه رسمی دست‌نخورده، اجرا می‌شود.
   - threshold استفاده‌شده و برچسب‌های پیش‌بینی fracture/triage در خروجی ثبت می‌شوند.
   - گزارش قبلی که threshold=0.5 داشت از این ناسازگاری آسیب ندیده است.

5. **`scripts/run_balanced_fold0.py` — اجرای آزمایش و گزارش پس از پایان**
   - قفل advisory برای جلوگیری از اجرای تکراری همین آزمایش؛ child نیز lock را نگه می‌دارد.
   - شروع فقط همین Fold 0 با تنظیمات مشخص؛ پشتیبانی از `--resume` و `--evaluate-only`.
   - پس از اتمام آموزش، `best.pt` روی validation فولد صفر با batch=1 و top3_mean/threshold=0.5 ارزیابی می‌شود.
   - نتیجه با baseline همان مطالعات مقایسه و در `comparison_to_baseline.json` ثبت می‌شود. گزارش توسعه‌ای است، نه نتیجه test یا پنج fold.

### تست‌های انجام‌شده تا این مرحله

فرمان:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

نتیجه: **۸ تست پاس شد**.

| تست | چه چیزی بررسی شد؟ | نتیجه |
|---|---|---|
| batch ratio/budget | نسبت مثبت در batch کامل و ناقص، حفظ طول epoch | پاس |
| patient balance | اختلاف تعداد برداشت از بیماران یک کلاس حداکثر یک است | پاس |
| negative rotation | طی کران محاسبه‌شده همه برش‌ها واقعاً انتخاب می‌شوند | پاس |
| deterministic resume | بازسازی ترتیب با شماره epoch و seed بدون تاریخچه RNG | پاس |
| invalid policies | رد نسبت/اندازه batch نامعتبر، کلاس تک‌حالته و عدم تطابق طول داده | پاس |
| real DataLoader | loader واقعی Torch در epochهای ۰، ۱ و ۶ دقیقاً ترتیب sampler را مصرف می‌کند | پاس |
| custom QWK threshold | تغییر مرز ۰٫۵ به ۰٫۲۵ و رفتار مساوی مرز، کلاس رسمی مورد انتظار را می‌دهد | پاس |
| high MLS rule | MLS بالا بدون خونریزی/شکستگی Urgent و با fracture، Critical می‌شود | پاس |

`py_compile` برای پنج فایل اجرایی تغییرکرده نیز بدون خطا گذشت.

**Audit روی داده واقعی بدون آموزش:** فهرست‌های Fold 0 خوانده شد؛ بیمار مشترک صفر بود. هر epoch تمام ۱۷۷ تصویر مثبت را پوشش می‌دهد. در epoch اول ۲۳۱۷ منفی یکتا از pool شامل ۳۳۴۱ منفی انتخاب می‌شود. سهم هر بیمار منفی ۲۱ یا ۲۲ برداشت و سهم هر بیمار مثبت ۴۶ یا ۴۷ برداشت است. کران پوشش همه منفی‌ها ۵ epoch است. تعداد برداشت با تعداد تصویر یکتا تفاوت دارد.

### دسترسی GPU و وضعیت اجرا

- در sandbox، `nvidia-smi` ناموفق و `torch.cuda.is_available()` برابر False بود.
- بررسی خواندنی خارج از sandbox موفق شد: Quadro RTX 8000، حافظه کل 49152 MiB و حدود 35866 MiB آزاد.
- runner قبلی PID=3735881 و آموزش Fold 1 با PID=3735886 در میزبان GPU فعال بودند؛ در sandbox این پردازش‌ها دیده نمی‌شدند. هیچ‌یک متوقف یا دوباره اجرا نشدند.
- **در زمان این ثبت اولیه، آموزش B25 هنوز شروع نشده است.** تست کوتاه یکپارچه آموزش و سپس شروع اجرای اصلی در ادامه همین گزارش ثبت می‌شود.

### معیار تصمیم و محدودیت‌های نتیجه

- مقایسه اصلی: AP مطالعه، حساسیت/ویژگی، TP/FP/FN/TN، QWK ایزوله و زمان اجرا. بهترشدن loss یا QWK به‌تنهایی کافی نیست.
- تنها پنج مطالعه مثبت در validation این fold وجود دارد؛ تغییر یک مورد می‌تواند معیارها را زیاد جابه‌جا کند. آزمایش Fold 0 توسعه‌ای است و بهبود باید بعداً در پروتکل بیمارمحور گسترده‌تر تأیید شود.
- verifier، HNM، threshold قابل‌انتخاب و calibration باید فقط از داده آموزش/تقسیم داخلی همان fold استفاده کنند. داده بدون annotation وارد YOLO loss نمی‌شود.
- initialization جدید پیش از ساخت P2 با seed=42 ثابت شده، اما baseline قدیمی قبل از initialization seed نمی‌کرد. بنابراین این مقایسه همه تفاوت‌های تصادفی initialization را حذف نمی‌کند؛ بهبود کوچک به کنترل با initialization/seed مشترک نیاز دارد.
- QWK این پروژه در ارزیابی fracture از ICH و MLS واقعی استفاده می‌کند؛ امتیاز کامل submission تیم نیست.

### قدم‌های بعدی این نوبت

1. اجرای کوتاه واقعی train/validation برای اطمینان از سازگاری sampler با Ultralytics و ذخیره checkpoint.
2. شروع آزمایش B25 در GPU و بررسی لاگ واقعی اولین epoch.
3. ثبت PID، مسیر لاگ، وضعیت، نتایج تست و دستورهای ادامه/مشاهده در همین فایل.
4. پس از پایان آموزش، runner به‌طور خودکار ارزیابی و مقایسه را ذخیره می‌کند؛ تا قبل از آن هیچ ادعایی درباره افزایش دقت نمی‌شود.

## تکمیل تست یکپارچه — 2026-09-12، پس از نیمه‌شب تهران

**پیاده‌سازی جدید:** `scripts/smoke_balanced_training.py` اضافه شد. این اسکریپت از تصاویر موجود و دارای annotation در train/validation فولد صفر، ۳۲ تصویر train (۸ مثبت، ۲۴ منفی) و ۸ تصویر val را انتخاب می‌کند؛ فهرست‌های موقت و وزن‌های smoke در پوشه موقت سیستم هستند. هیچ تصویر جدید یا label آموزشی تولید نمی‌کند. فایل‌های موقت smoke پس از اتمام پاک می‌شوند؛ Ultralytics ممکن است cache قابل‌بازسازی برچسب‌ها را بازتولید کند و خود labelها تغییر نمی‌کنند.

**فرمان اجرا:**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python scripts/smoke_balanced_training.py
```

**نتیجه واقعی: پاس، exit code=0.** آموزش/validation روی CPU در imgsz=128 و batch=16 اجرا شد. پس از epoch اول، وجود optimizer و epoch=0 در last.pt بررسی شد. سپس یک توقف عمدی قبل از اجرای epoch دوم ایجاد شد و دستور معمول `--resume`، بدون دادن گزینه sampler، آموزش را تا epoch دوم ادامه داد.

موارد تأییدشده:

- در هر epoch دقیقاً دو batch و ۸ برداشت مثبت/۲۴ برداشت منفی مصرف شد.
- policy با نسبت 0.25 در best.pt نهایی موجود بود.
- hash ترتیب نمونه‌گیری epoch دوم پیش و پس از resume یکسان بود. این تأیید مربوط به sampler است؛ بازپخش بیت‌به‌بیت همه augmentationها/optimizer ادعا نمی‌شود.
- lossها عدد متناهی داشتند، train/validation و ذخیره/بارگذاری checkpoint موفق شدند.
- metrics تشخیص در این تست کوچک صفر بود؛ این تست صرفاً اتصال نرم‌افزار است و هیچ شاهدی برای کیفیت مدل محسوب نمی‌شود.
- پیام نهایی: `SMOKE PASSED: CPU train/val, checkpoint, interrupted resume, identical sampling schedule; 2 epochs, 32 train / 8 val images at 128px.`

**تغییر تکمیلی:** trainer نسخه sampler را با policy تطبیق می‌دهد؛ ناسازگاری نسخه در resume خطا می‌دهد. `scripts/run_balanced_fold0.py` اکنون شروع، خطای subprocess، پایان آموزش، نتایج ارزیابی و مقایسه را خودکار به همین `agent.md` اضافه می‌کند تا نتیجه آینده نیز ثبت شود. README فرمان‌های start/resume/evaluate و تست‌ها را توضیح می‌دهد.

**اجرای اصلی بعد از این تست:** فرمان شروع فقط روی میزبان دارای GPU اجرا می‌شود؛ PID و تأیید اولین batch/epoch در ثبت بعدی می‌آید.

### ثبت خودکار B25 — شروع runner — 2026-09-12T00:05:35+03:30

PID=3963096؛ device=0؛ resume=False؛ evaluate_only=False. مسیر اجرا: `/mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12/outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25`. فرضیه: افزایش سهم مثبت‌ها با توازن بیمار، حساسیت مطالعه را بهتر کند؛ هنوز نتیجه آموزشی جدیدی موجود نیست.

## تأیید شروع GPU و تست‌های نهایی — 2026-09-12، ساعت 00:07 تهران

### ماندگاری فرایند و اجرای واقعی

- تلاش اول با `nohup` تنها، PID=3961134 برگرداند ولی در بررسی بعدی فرایند وجود نداشت و لاگ خالی بود؛ این تلاش شروع موفق آموزش محسوب نشد.
- تلاش دوم با `setsid nohup` در session مستقل موفق بود. PID runner برابر **3963096**، PPID برابر **1** و SID برابر **3963096** است. PID فرایند آموزش **3963158** است. این اجرا به session ترمینال اولیه وابسته نیست.
- فرمان واقعی:

```bash
setsid nohup .venv/bin/python -u scripts/run_balanced_fold0.py --device 0 \
  >> outputs/balanced_fold0.log 2>&1 < /dev/null &
```

- تست AMP در GPU پاس شد. لاگ، شروع epoch اول و پیشرفت تا حداقل batch **76/220** را نشان داد. lossها متناهی بودند؛ در آن لحظه box_loss≈2.656، cls_loss≈16.64، l1_loss≈0.02094 بود. این مقادیر ابتدای آموزش‌اند و معیار بهبود دقت نیستند.
- `nvidia-smi` برای فرایند جدید **9824 MiB** و برای Fold 1 قبلی **12718 MiB** نشان داد. آموزش قبلی با PID=3735886 همچنان فعال بود.
- ثبت واقعی sampler در loader: ۳۵۱۸ برداشت، ۲۲۰ batch، ۸۸۰ مثبت، ۲۶۳۸ منفی؛ ۱۷۷ مثبت یکتا و ۲۳۱۷ منفی یکتا. سیاست در `sampling_policy.json` و آمار epoch در `sampling_epochs.jsonl` همان پوشه ذخیره شده است.
- hash اندیس‌های epoch صفر در loader واقعی: `e13bca86092b5fec05518d72099ebaf673a4f61659dcb8fce30a326cd029b9c8`. اندیس‌ها نسبت به ترتیب dataset مرتب‌شده Ultralytics هستند؛ audit اولیه قبل از ساخت loader نسبت به ترتیب فهرست train بود و hash اندیس آن مستقیماً قابل مقایسه نیست. آمار سهم/پوشش برابر بود.

### تست‌های تکمیلی و نتیجه

سه تست به `tests/test_balanced_sampling.py` اضافه شد، چون جلوگیری از نشت بیمار و مصرف داده بدون annotation جزء الزامات اصلی کاربر است:

1. ورود دو تصویر متفاوت از یک بیمار به train/val → خطای `Patient leakage`؛ پاس.
2. ورود تصویر با برچسب split=test به train → رد شدن؛ پاس.
3. تصویر موجود ولی بدون JSON annotation → رد شدن، حتی اگر label خالی و شمارش باکس صفر باشد؛ پاس.

کل suite پس از این اضافه‌ها: **۱۱ تست پاس شد**. کامپایل فایل runner و trainer نیز موفق بود. تست کوتاه واقعی CPU و resume در بخش قبل ثبت شده و بدون تغییر منطق sampler دوباره اجرا نشده است.

**تست جلوگیری از اجرای تکراری:** هنگام فعال بودن runner، فرمان `python scripts/run_balanced_fold0.py --device 0` دوباره فراخوانی شد. خروج مورد انتظار با کد 1 و پیام `This experiment is already running (lock held)` رخ داد؛ هیچ آموزش دومی از B25 شروع نشد. این کد خروج، نتیجه موفق guard است و خطای آموزش اصلی نیست.

### نسخه کد در زمان شروع آزمایش

| فایل | SHA256 |
|---|---|
| `src/patient_balanced_sampling.py` | `f881924f8e26b0721fb96db8d325e449b7ca07b1deb750af634b1bdef92fee89` |
| `src/balanced_yolo_trainer.py` | `1bca06979b89495dfc781bc41bef266dda0153e8dc61d7d35b801068ed9a9298` |
| `src/train_yolo26s_p2.py` | `af71b116f485940767eb637a45a6af76db0e33de23502bfc6bf8b2055d9ba87f` |
| `src/evaluate_fold_qwk.py` | `94af0a94fe3324a5fec6bb12c35de576dc6cceb6167c183c50af3af38ca04daf` |
| `scripts/run_balanced_fold0.py` | `1205fbe462898195867303f315d86b4312ceb1a700306db605aef6a207eb8b48` |
| `scripts/smoke_balanced_training.py` | `8e9fd2fe20b07cc59857a36607f6ea566954dd3a6707f7faacac1c8a3146a77d` |
| `tests/test_balanced_sampling.py` | `95c2c9e974b8be99888950148fad1822647642600b759ef87ddfc30fa4446c6b` |
| `tests/test_fold_qwk_threshold.py` | `5637f9f760241b53aa34f57865d0c8e34554f120107376fd1390877296ed9486` |

### دستورهای مشاهده و ادامه

اجرای جدید اکنون فعال است؛ فرمان شروع را دوباره اجرا نکنید. برای مشاهده:

```bash
tail -f outputs/balanced_fold0.log
```

فقط در صورت قطع واقعی آموزش و وجود last.pt دارای optimizer:

```bash
setsid nohup .venv/bin/python -u scripts/run_balanced_fold0.py --device 0 --resume \
  >> outputs/balanced_fold0.log 2>&1 < /dev/null &
```

پس از اتمام، خروجی‌های مورد انتظار در `outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/` عبارت‌اند از `weights/best.pt`، `study_evaluation/metrics.json`، `study_evaluation/study_predictions.csv`، `comparison_to_baseline.json` و `experiment_complete.json`. این‌ها در زمان شروع همگی ساخته نشده‌اند؛ نتیجه و پایان واقعی به‌صورت خودکار پایین همین گزارش ثبت می‌شود.

## تأیید اولین epoch کامل GPU — 2026-09-12، ساعت 00:08:45 تهران

**وضعیت واقعی هنگام تحویل:** epoch اول با train و validation تمام شده و آموزش وارد epoch دوم شده است (در آخرین مشاهده batch 48/220 از epoch 2/150). هیچ Traceback یا Error در بررسی لاگ مشاهده نشد. `last.pt`، `best.pt` و `epoch0.pt` هرکدام با اندازه 78,508,935 بایت ساخته شده‌اند؛ برخلاف checkpoint نهایی stripped، این‌ها هنوز وضعیت آموزش دارند.

نتایج ثبت‌شده در `results.csv` پس از epoch اول:

| معیار | مقدار |
|---|---:|
| زمان تجمعی آموزش/ارزیابی تا ثبت سطر اول | 143.456 ثانیه |
| train box_loss | 2.49061 |
| train cls_loss | 8.42860 |
| train l1_loss | 0.02033 |
| validation precision باکس | 0.28743 |
| validation recall باکس | 0.25316 |
| validation mAP50 | 0.17682 |
| validation mAP50–95 | 0.05018 |

این صرفاً تأیید اجرای سالم یک epoch است؛ مقایسه با best epoch 42 baseline در این مرحله عادلانه نیست. ارزیابی نهایی study-level و QWK پس از پایان آموزش توسط runner انجام می‌شود. اجرای baseline قبلی متوقف نشده و آزمایش فعلی در پوشه مستقل است.

### ایده‌های بعدی، مشروط به نتیجه B25

- اگر حساسیت بیشتر و FP قابل‌قبول شد: همین سیاست در سایر foldها بررسی شود؛ این مرحله هنوز اجرا نشده است.
- اگر FNهای بدون پیشنهاد مناسب باقی ماندند: inference در 1024 به‌عنوان آزمایش جداگانه برای بررسی نمایش جزئیات و پوشش کاندیداها؛ P2 به‌تنهایی تضمین نمی‌کند خط باریک شکستگی قابل‌تشخیص باشد.
- اگر پیشنهادهای منطبق ولی کم‌امتیاز و FPهای مشابه ساختار طبیعی زیاد بودند: verifier کوچک با crop محلی و اطراف آن؛ انتخاب/آموزش فقط از train و تقسیم داخلی همان fold.
- اگر پس از افزایش حساسیت FP افزایش یافت: HNM با منفی‌های تأییدشده از train همان fold، مستقل از تغییر نسبت نمونه‌گیری.
- مقایسه m-P2، SAHI، window مکمل یا MIL تنها با شواهد خطای باقی‌مانده و آزمایش جداگانه توجیه می‌شود. هیچ‌کدام در اجرای فعلی فعال نشده‌اند و سود مشخصی برای آن‌ها ادعا نمی‌شود.

### ثبت خودکار B25 — پایان آموزش — 2026-09-12T03:18:47+03:30

فرایند آموزش با کد صفر تمام شد؛ ارزیابی مستقل از loader آموزش روی validation همان Fold 0 آغاز می‌شود. این رخداد هنوز به معنی بهبود دقت نیست.

### ثبت خودکار B25 — پایان ارزیابی validation — 2026-09-12T03:19:14+03:30

خروجی واقعی معیارهای مطالعه و QWK ایزوله (ICH/MLS واقعی):

```json
{
  "fracture": {
    "threshold": 0.5,
    "tp": 2,
    "fp": 4,
    "fn": 3,
    "tn": 24,
    "sensitivity": 0.4,
    "specificity": 0.8571428571428571,
    "precision": 0.3333333333333333,
    "accuracy": 0.7878787878787878,
    "binary_qwk": 0.2376237623762376,
    "study_pr_auc": 0.6334267040149393,
    "study_roc_auc": 0.85
  },
  "isolated_qwk": 0.9504504504504505
}
```

این امتیاز نتیجه Fold 0 است؛ پنج‌فولدی یا test نهایی نیست.

### ثبت خودکار B25 — مقایسه با baseline — 2026-09-12T03:19:14+03:30

نتیجه مقایسه در `outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/comparison_to_baseline.json` ذخیره شد.

```json
{
  "protocol": "exploratory fold-0 development ablation; not final test or five-fold evidence",
  "baseline_report": "/mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12/outputs/yolo26s_p2_hu800_ww1600_fold0/study_evaluation/metrics.json",
  "experiment_report": "/mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12/outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/study_evaluation/metrics.json",
  "metrics": {
    "tp": {
      "baseline": 2,
      "balanced25": 2
    },
    "fp": {
      "baseline": 0,
      "balanced25": 4
    },
    "fn": {
      "baseline": 3,
      "balanced25": 3
    },
    "tn": {
      "baseline": 28,
      "balanced25": 24
    },
    "sensitivity": {
      "baseline": 0.4,
      "balanced25": 0.4
    },
    "specificity": {
      "baseline": 1.0,
      "balanced25": 0.8571428571428571
    },
    "precision": {
      "baseline": 1.0,
      "balanced25": 0.3333333333333333
    },
    "study_pr_auc": {
      "baseline": 0.5366507177033493,
      "balanced25": 0.6334267040149393
    },
    "study_roc_auc": {
      "baseline": 0.6928571428571428,
      "balanced25": 0.85
    }
  },
  "isolated_triage_qwk": {
    "baseline": 0.9751693002257337,
    "balanced25": 0.9504504504504505
  },
  "initialization_note": "New P2 initialization is explicitly seeded; historical baseline did not seed before initialization. Confirm small gains with matched-seed controls."
}
```

## دستورهای مشاهده نتایج دو اجرای کامل — 2026-09-12

در این تاریخ بررسی شد که هر پنج اجرای baseline K-fold کامل و وزن‌های نهایی آن‌ها stripped شده‌اند. بهترین epochهای گزارش‌شده در لاگ‌ها برای foldهای ۰ تا ۴ به‌ترتیب ۴۲، ۹۹، ۶۱، ۲۰ و ۵۹ است. آزمایش Fold 0 با نمونه‌گیری balanced25 نیز در epoch ۳۶ بهترین نتیجه باکس را ثبت کرده، در epoch ۶۶ با early stopping تمام شده و فایل `experiment_complete.json` دارد. پوشه قدیمی `outputs/yolo26s_p2_hu800_ww1600` در این مقایسه وارد نمی‌شود.

### مشاهده خروجی‌های آموزش K-fold

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
source .venv/bin/activate

for FOLD_ID in 0 1 2 3 4; do
  echo "===== FOLD ${FOLD_ID} ====="
  tail -n 12 "outputs/kfold_logs/fold_${FOLD_ID}.log"
done
```

نمایش بهترین epoch و معیارهای detector از `results.csv` بدون inference جدید:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
import pandas as pd

for fold in range(5):
    path = Path(f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/results.csv")
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    best = table.loc[table["metrics/mAP50-95(B)"].idxmax()]
    print(
        f"fold={fold} epochs={len(table)} best_epoch={int(best['epoch'])} "
        f"P={best['metrics/precision(B)']:.5f} R={best['metrics/recall(B)']:.5f} "
        f"mAP50={best['metrics/mAP50(B)']:.5f} "
        f"mAP50-95={best['metrics/mAP50-95(B)']:.5f}"
    )
PY
```

برای ساخت ارزیابی study-level/QWK روی validation نگه‌داشته‌شده هر Fold، بدون استفاده از test ثابت:

```bash
for FOLD_ID in 0 1 2 3 4; do
  RUN_DIR="outputs/yolo26s_p2_hu800_ww1600_fold${FOLD_ID}"
  mkdir -p "${RUN_DIR}/study_evaluation"
  PYTHONPATH=src python src/evaluate_fold_qwk.py \
    --fold "${FOLD_ID}" \
    --weights "${RUN_DIR}/weights/best.pt" \
    --output "${RUN_DIR}/study_evaluation" \
    --aggregator top3_mean --threshold 0.5 \
    --device 0 --batch 8 --imgsz 768 \
    | tee "${RUN_DIR}/study_evaluation/evaluation_console.log"
done
```

نمایش خلاصه pooled پنج Fold بعد از اجرای فرمان بالا:

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score

tables = []
for fold in range(5):
    path = Path(f"outputs/yolo26s_p2_hu800_ww1600_fold{fold}/study_evaluation/study_predictions.csv")
    frame = pd.read_csv(path, dtype={"study_id": str})
    frame["fold"] = fold
    tables.append(frame)
oof = pd.concat(tables, ignore_index=True)
if oof.study_id.duplicated().any():
    raise RuntimeError("A study appears in more than one validation fold")
y = oof.fracture_true.astype(bool).to_numpy()
s = oof.top3_mean.to_numpy()
p = s >= 0.5
tn, fp, fn, tp = confusion_matrix(y, p, labels=[False, True]).ravel()
summary = {
    "studies": len(oof), "positive_studies": int(y.sum()),
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "sensitivity": float(tp / (tp + fn)),
    "specificity": float(tn / (tn + fp)),
    "precision": float(tp / (tp + fp)) if tp + fp else None,
    "study_pr_auc": float(average_precision_score(y, s)),
    "study_roc_auc": float(roc_auc_score(y, s)),
    "binary_qwk": float(cohen_kappa_score(y.astype(int), p.astype(int), weights="quadratic")),
}
Path("outputs/kfold_study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
oof.to_csv("outputs/kfold_oof_study_predictions.csv", index=False)
print(json.dumps(summary, indent=2))
PY
```

این pooled summary مربوط به ۱۵۵ بیمار train+validation در OOF است؛ ۲۸ بیمار test ثابت را مصرف نمی‌کند. QWK ایزوله سه‌کلاسه هر Fold در `metrics.json` موجود است، اما میانگین ساده پنج QWK جای QWK pooled یا QWK کامل تیم را نمی‌گیرد.

### مشاهده آزمایش balanced25 و مقایسه با baseline Fold 0

```bash
python -m json.tool \
  outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/study_evaluation/metrics.json

python -m json.tool \
  outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/comparison_to_baseline.json
```

بهترین epoch و معیارهای detector آزمایش:

```bash
PYTHONPATH=src python - <<'PY'
import pandas as pd
p = "outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/results.csv"
d = pd.read_csv(p)
d.columns = d.columns.str.strip()
b = d.loc[d["metrics/mAP50-95(B)"].idxmax()]
print(f"epochs={len(d)} best_epoch={int(b['epoch'])}")
print(b[["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]].to_string())
PY
```

فایل‌های تصویری مهم برای بازکردن در IDE، برای هر Fold یا balanced25: `results.png`، `BoxPR_curve.png`، `BoxF1_curve.png`، `confusion_matrix.png`، `confusion_matrix_normalized.png`، `val_batch*_labels.jpg` و `val_batch*_pred.jpg`. نمونه فرمان محیط گرافیکی:

```bash
xdg-open outputs/yolo26s_p2_hu800_ww1600_fold0/results.png
xdg-open outputs/yolo26s_p2_hu800_ww1600_fold0_balanced25/results.png
```

نتیجه واقعی balanced25 که از قبل ذخیره شده: mAP50-95 باکس از 0.09725 به 0.13166 و PR-AUC مطالعه از 0.53665 به 0.63343 رسید، ولی sensitivity در 0.4 ثابت ماند، FP از 0 به 4 افزایش یافت و QWK ایزوله از 0.97517 به 0.95045 افت کرد. بنابراین این آزمایش در آستانه 0.5 جایگزین baseline نشده است.

## ارزیابی مستقل QWK شکستگی روی test ثابت — 2026-09-12

### تعریف معیار و تصمیم پیش از inference

درخواست کاربر: محاسبه QWK آموزش منتخب روی test، مستقل از دو task دیگر. معیار مناسب در این دامنه **Binary QWK در سطح Study** است: برچسب واقعی `SkullFracture` در برابر تصمیم شکستگی مدل. هیچ مقدار ICH volume، MLS یا triage class در محاسبه خوانده نمی‌شود. در مسئله دوکلاسه، quadratic weighted kappa با Cohen kappa معمولی برابر است.

مدل منتخب پیش از بازکردن test، ensemble پنج `best.pt` متعلق به K-fold تعیین شد. balanced25 انتخاب نشد، چون با وجود mAP/PR-AUC بهتر روی development Fold 0، در threshold=0.5 چهار FP بیشتر و QWK ایزوله پایین‌تری داشت. انتخاب یک fold بر اساس نتیجه test ممنوع در نظر گرفته شد.

پروتکل اصلی ثابت:

1. برای هر یک از ۷۵۸ slice تست، بیشترین confidence با کف proposal=0.001 از هر پنج مدل گرفته شود.
2. پنج confidence همان slice میانگین گرفته شود.
3. در هر Study میانگین سه slice با بالاترین score محاسبه شود.
4. threshold بدون sweep روی test برابر 0.5 باشد.
5. diagnostic تک‌مدل‌ها ذخیره شود ولی برای انتخاب مدل/threshold استفاده نشود.

### پیاده‌سازی

- `src/evaluate_fracture_test_qwk.py`: کنترل دقیق fixed test، عدم هم‌پوشانی بیمار، تطبیق ground truth با box annotation و metadata، inference پنج وزن، ensemble، معیارها، hash وزن‌ها و bootstrap خوشه‌ای بیمار.
- `tests/test_fracture_test_qwk.py`: آزمون top3_mean، QWK کامل دودویی و bootstrap بیمارمحور.
- کل suite بعد از تغییر: **۱۴ تست پاس**؛ `py_compile` نیز پاس شد.
- پس از نخستین اجرای موفق، guard اضافه شد که اگر `outputs/fixed_test_fracture_kfold_ensemble/metrics.json` موجود باشد از overwrite یا اجرای تکراری test جلوگیری کند.

### جمعیت ارزیابی

- ۷۵۸ تصویر annotation-backed
- ۲۸ بیمار، بدون هم‌پوشانی با development
- ۲۹ Study
- ۴ بیمار مثبت و ۴ Study مثبت
- hash فهرست test: `4ccb638ddc7d2daeec1d7175b039a3dad9221678f0ddbfa34ea37b135e7433e1`

### نتیجه اصلی مستقل از ICH/MLS

| معیار | مقدار |
|---|---:|
| TP / FP / FN / TN | 0 / 0 / 4 / 25 |
| Sensitivity | 0.0000 |
| Specificity | 1.0000 |
| Accuracy | 0.8621 |
| **Binary QWK** | **0.0000** |
| Study PR-AUC | 0.39764 |
| Study ROC-AUC | 0.70000 |
| bootstrap 95% CI برای QWK | [0.0, 0.0]؛ چون تصمیم همه‌منفی در تمام بازنمونه‌گیری‌ها ثابت است |

Accuracy بالا از عدم‌توازن می‌آید و نشان‌دهنده تشخیص درست شکستگی نیست. PR-AUC پایه تقریبی با چهار مثبت از ۲۹ Study برابر 0.1379 است؛ PR-AUC مدل بالاتر از prevalence است و نشان می‌دهد scoreها مقداری اطلاعات رتبه‌بندی دارند، اما threshold رسمی 0.5 هیچ مثبت واقعی را عبور نداده است.

امتیاز چهار Study مثبت:

| Study | positive slices | ensemble top3_mean | ensemble max |
|---|---:|---:|---:|
| 6871 | 12 | 0.203024 | 0.252672 |
| 1196 | 18 | 0.193528 | 0.215187 |
| 3356 | 4 | 0.015135 | 0.027199 |
| 1839 | 3 | 0.005451 | 0.009718 |

بالاترین Study کل test یک نمونه منفی با ID داخلی 271922 و top3_mean=0.359177 بود. این مشاهده فقط تحلیل خطا است؛ آستانه نباید از روی همین test پایین آورده شود.

### artifactها و مشاهده

- `outputs/fixed_test_fracture_kfold_ensemble/metrics.json`
- `outputs/fixed_test_fracture_kfold_ensemble/study_predictions.csv`
- `outputs/fixed_test_fracture_kfold_ensemble/slice_predictions.csv`

```bash
.venv/bin/python -m json.tool outputs/fixed_test_fracture_kfold_ensemble/metrics.json
column -s, -t < outputs/fixed_test_fracture_kfold_ensemble/study_predictions.csv | less -S
```

### تفسیر و محدودیت

این QWK مستقل fracture است و QWK رسمی سه‌کلاسه مسابقه نیست. عدد رسمی سه‌کلاسه بدون پیش‌بینی واقعی ICH و MLS قابل محاسبه مستقل نیست. test تنها چهار Study مثبت دارد، بنابراین عدم‌قطعیت درباره ranking زیاد است؛ بازه QWK در این نقطه عملیاتی صفر مانده چون همه پیش‌بینی‌ها منفی‌اند. نتیجه test نباید برای انتخاب fold، aggregator یا threshold استفاده شود. قدم بعدی صحیح، تحلیل OOF/development و ساخت calibration یا بهبود مدل فقط روی development است؛ هر نسخه جدید باید روی test جدید/پنهان سنجیده شود، نه با تکرار انتخاب روی این fixed test.

## انتخاب آستانه Binary QWK از OOF و اعمال فریز‌شده — 2026-09-12

### ایده و دلیل

مشاهده شد که مدل در test رتبه‌بندی کاملاً تصادفی ندارد (`PR-AUC=0.39764` در برابر prevalence برابر 0.1379)، اما threshold ثابت 0.5 همه Studyها را منفی می‌کند. پایین‌آوردن آستانه بر اساس چهار Positive test مصداق tune روی test و biased است. بنابراین آستانه فقط از predictionهای out-of-fold پنج مدل انتخاب شد:

`t* = argmax_t Binary-QWK_OOF(t)`

واحد انتخاب Study و score همان `top3_mean` هر مدل است. candidateها تمام decision boundaryهای یکتای OOF را پوشش می‌دهند. tie-break از پیش در کد ثابت شد: QWK بیشتر، سپس specificity بیشتر، sensitivity بیشتر و در پایان فاصله کمتر با 0.5. ICH و MLS در هیچ مرحله‌ای خوانده نمی‌شوند.

برای جلوگیری از گزارش خوش‌بینانه‌ی عملکرد threshold، یک ارزیابی cross-fitted نیز انجام شد: برای هر Fold، threshold فقط روی چهار Fold دیگر انتخاب و روی Fold کنارگذاشته‌شده اعمال شد؛ سپس تصمیم‌های هر پنج Fold pool شدند. آستانه‌ی deployment پس از این ارزیابی از کل OOF فریز شد.

از آنجا که توزیع confidence مدل‌های منفرد با ensemble میانگین‌گیری‌شده یکسان نیست، در test برای هر مدل همان threshold مشترک OOF اعمال شد و تصمیم ensemble برابر رأی اکثریت (حداقل ۳ رأی از ۵ مدل) تعریف شد. هیچ threshold sweep، انتخاب fold یا انتخاب aggregator روی test انجام نشد.

### پیاده‌سازی و کنترل‌ها

- `src/select_oof_fracture_threshold.py`: بارگذاری دقیق پنج فایل OOF، تطبیق ۱۶۹ Study با تمام development manifest، کنترل عدم تکرار Study و عدم پخش یک بیمار در چند Fold، sweep فقط روی OOF، cross-fitting، freeze و اعمال یک‌باره به predictionهای ذخیره‌شده‌ی test.
- `tests/test_oof_fracture_threshold.py`: پوشش candidateها، جداسازی کامل، metric از decision ثابت، استقلال held-out در cross-fit و bootstrap تصمیم ثابت.
- کل suite: **۱۹ تست پاس**؛ `py_compile` نیز پاس است.
- guard مانع overwrite شدن `threshold_report.json` می‌شود.
- خروجی نخست پیش از افزودن reference آستانه 0.5 و bootstrap، به‌صورت recoverable به `outputs/oof_fracture_threshold_v1_initial` منتقل شد. خروجی کامل و authoritative در `outputs/oof_fracture_threshold_v1` است.

### نتیجه OOF

جمعیت OOF شامل ۱۶۹ Study، ۱۵۵ بیمار و ۲۴ Study مثبت است.

| روش | TP / FP / FN / TN | Sens. | Spec. | Precision | Binary QWK | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| OOF با threshold=0.5 | 6 / 0 / 18 / 145 | 0.2500 | 1.0000 | 1.0000 | 0.36386 | 0.55659 | 0.73204 |
| کل OOF در threshold منتخب؛ optimistic | 9 / 1 / 15 / 144 | 0.3750 | 0.9931 | 0.9000 | 0.48652 | 0.55659 | 0.73204 |
| OOF cross-fitted؛ برآورد محافظه‌کارانه | 8 / 7 / 16 / 138 | 0.3333 | 0.9517 | 0.5333 | 0.33793 | 0.55659 | 0.73204 |

آستانه‌ی نهایی فریز‌شده از کل OOF برابر **0.28888921362037456** است. thresholdهای cross-fitted برای Foldهای 0 تا 4 به‌ترتیب `0.216992`، `0.288889`، `0.420364`، `0.288889` و `0.180534` بودند. پراکندگی آن‌ها نشان می‌دهد calibration بین Foldها پایدار نیست؛ به همین علت عدد apparent کل OOF نباید برآورد بدون bias تلقی شود و عدد cross-fitted برای قضاوت threshold معتبرتر است.

### اعمال ثانویه روی fixed test

با threshold فریز‌شده و رأی اکثریت ۵ مدل:

| معیار | مقدار |
|---|---:|
| TP / FP / FN / TN | 1 / 1 / 3 / 24 |
| Sensitivity | 0.2500 |
| Specificity | 0.9600 |
| Precision | 0.5000 |
| Accuracy | 0.8621 |
| **Binary QWK** | **0.26582** |
| PR-AUC رأی مدل‌ها | 0.36063 |
| ROC-AUC رأی مدل‌ها | 0.72500 |
| PR-AUC score پیوسته ensemble اولیه | 0.39764 |

Study مثبت `1196` سه رأی مثبت گرفت و TP شد. Study مثبت `6871` فقط دو رأی گرفت و منفی ماند. Study منفی `271922` سه رأی مثبت گرفت و FP شد. دو Positive دیگر هیچ رأیی نگرفتند. بازه bootstrap خوشه‌ای/طبقه‌بندی‌شده‌ی بیمار برای QWK برابر `[-0.1053, 0.7130]`، sensitivity برابر `[0, 0.75]` و specificity برابر `[0.875, 1]` است؛ پهنای بسیار زیاد بازه‌ها مستقیماً ناشی از تنها چهار Positive test است.

این نتیجه فرضیه‌ی نامناسب‌بودن 0.5 را تا حدی پشتیبانی می‌کند، اما اثبات قطعی بهبود generalization نیست: QWK از 0 به 0.2658 رسید، در حالی که cross-fitted OOF فقط 0.3379 است و calibration Foldها ناپایدار است. مهم‌تر اینکه test قبلاً با threshold=0.5 مشاهده شده بود؛ پس با وجود عدم sweep روی test، این خروجی باید **secondary/post-hoc** نامیده شود و برای ادعای نهایی به test پنهان جدید نیاز است.

### جمع‌بندی روش‌شناختی و اولویت مرحله بعد

سه عدد زیر نباید با یکدیگر جایگزین شوند:

- `OOF QWK @ 0.5 = 0.36386`: عملکرد همان operating point اولیه روی OOF.
- `OOF QWK @ selected threshold = 0.48652`: نتیجه apparent که threshold روی کل OOF انتخاب و روی همان داده گزارش شده و خوش‌بینانه است.
- `Cross-fitted OOF QWK = 0.33793`: برآورد محافظه‌کارانه‌تر برای generalization لایه‌ی threshold و عدد اصلی برای قضاوت این مرحله.

عبارت مناسب برای نتیجه test: «آستانه‌ی استخراج‌شده از OOF در تحلیل ثانویه، Binary QWK مجموعه test ثابت را از 0 به 0.266 رساند؛ اما چون test قبلاً با آستانه 0.5 مشاهده شده بود، این نتیجه post-hoc است و improvement مستقل محسوب نمی‌شود.» بازه bootstrap برابر `[-0.1053, 0.7130]` نیز نشان می‌دهد با فقط چهار Study مثبت، point estimate برابر 0.26582 عدم‌قطعیت بسیار زیادی دارد.

تعریف metricها باید صریح بماند:

- QWK، sensitivity، specificity، precision و confusion matrix test مربوط به **تصمیم majority vote** با حداقل سه رأی از پنج مدل هستند.
- `PR-AUC=0.36063` و `ROC-AUC=0.725` مربوط به score پیوسته‌ی `vote_fraction` همین ensemble رأی‌گیری‌اند.
- `PR-AUC=0.39764` و `ROC-AUC=0.700` مربوط به **continuous mean-confidence ensemble score اولیه** هستند و نباید به‌عنوان PR-AUC تصمیم majority vote معرفی شوند.

نتیجه عملی: threshold یکی از مشکلات بود، ولی bottleneck اصلی detector همچنان recall/evidence quality است؛ حتی پس از اصلاح operating point، سه مورد از چهار Positive test برابر FN مانده‌اند. بنابراین threshold tuning بیشتر فعلاً اولویت ندارد. آزمایش بعدی باید audit سه FN (`1839`، `3356`، `6871`) باشد و در سطح slice مشخص کند:

1. آیا هر مدل اصلاً در محل ground-truth fracture proposal هم‌پوشان تولید کرده است؟
2. اگر proposal وجود دارد، IoU و confidence آن چقدر است و در کدام مرحله‌ی pooling/voting حذف می‌شود؟
3. اگر proposal مناسب وجود ندارد، ابتدا inference با `imgsz=1024` و سپس بهبود detector بررسی شود؛ اگر proposal مناسب با score پایین وجود دارد، calibration یا verifier ارزش آزمون دارد.

برای Binary fracture analysis آستانه 0.288889 قابل استفاده است. برای rule رسمی مبتنی بر probability آستانه 0.5 نباید raw score را مستقیماً با این عدد جایگزین کرد؛ مسیر تمیزتر، یادگیری calibration فقط از OOF (`raw detector score -> calibrated fracture probability`) و سپس اعمال threshold رسمی 0.5 است. نوع calibrator و ارزیابی آن باید با cross-fitting انتخاب شود و test در این انتخاب نقشی نداشته باشد.

## Audit سه FN و یک FP در fixed test — 2026-09-12

### هدف و محدودیت

به‌جای شروع کورکورانه‌ی آموزش، منشأ خطاهای decision rule فریز‌شده بررسی شد. چهار Study خطادار عبارت‌اند از FNهای `1839`، `3356` و `6871` و FP با شناسه `271922`. این test قبلاً مشاهده شده است؛ تحلیل حاضر فقط برای تشخیص failure mode است و انتخاب مدل/تنظیمات بر اساس نتیجه همین چهار مورد مجاز نیست.

### پیاده‌سازی

- فایل `src/audit_fracture_test_errors.py` ساخته شد.
- پنج `best.pt` روی ۸۰ slice متعلق به چهار Study با `imgsz=768`، `conf=0.001`، `NMS IoU=0.7` و `max_det=300` دوباره inference شدند.
- GT از labelهای YOLO به مختصات pixel تبدیل و هر GT با تمام proposalهای post-NMS در آستانه‌های IoU برابر 0.30 و 0.50 مقایسه شد.
- scoreهای Study دوباره محاسبه و با predictionهای ذخیره‌شده تطبیق داده شدند؛ اختلاف عددی معناداری وجود نداشت.
- ۱۱۰ overlay تولید شد: GT سبز و حداکثر ده proposal برتر به‌علاوه بهترین match هر GT به رنگ قرمز.
- سه تست جدید برای IoU، تبدیل YOLO و حفظ confidence در matching اضافه شد. کل suite اکنون **۲۲ تست پاس** است و `py_compile` نیز پاس شد.

### نتایج دقیق

**Study 1839 — proposal failure کامل:** سه GT box دارد. در مجموع ۱۵ ارزیابی GT/model، هیچ proposal با IoU حتی 0.30 وجود نداشت. max confidence کلی مدل‌ها بین 0.0046 و 0.0374 بود، ولی هیچ‌کدام نزدیک GT نبود. calibration، threshold یا verifier مبتنی بر proposal این مورد را بازیابی نمی‌کند.

**Study 3356 — instability و confidence پایین:** شش GT box دارد. Foldهای 0، 1 و 4 به‌ترتیب 2، 1 و 3 GT را در IoU>=0.50 پیدا کردند؛ Foldهای 2 و 3 هیچ localization روی GT نداشتند. بهترین IoU برابر 0.7600 و بیشترین confidence proposal منطبق با GT فقط 0.1271 بود. این مورد هم کمبود evidence و هم instability بین Foldها را نشان می‌دهد.

**Study 6871 — localization موجود ولی score صحیح بسیار پایین:** سیزده GT box دارد. تمام Foldها localization با IoU>=0.50 دارند: تعداد matchها برای Foldهای 0 تا 4 برابر 6، 1، 3، 1 و 9 است؛ بهترین IoU برابر 0.8913 است. با این حال بیشترین confidence منطبق با GT فقط 0.0378 است. Foldهای 1 و 4 در Study-level رأی مثبت داده‌اند (`top3_mean=0.4362` و `0.3765`)، اما scoreهای پرقدرت الزاماً proposal صحیح fracture نیستند. بنابراین تصمیم مثبت آن دو مدل evidence-localized قابل اتکایی برای GT نیست.

**Study 271922 — FP روی نواحی غیرجمجمه‌ای/مرزی:** Study منفی و بدون GT است. Foldهای 0، 2 و 3 با `top3_mean=0.7864`، `0.3445` و `0.6428` مثبت می‌شوند؛ max confidence آن‌ها 0.8473، 0.3778 و 0.6893 است. بررسی overlay نشان می‌دهد بسیاری از proposalهای پرامتیاز در نیمه خارج جمجمه، مرز field-of-view و artifact/تجهیزات خارجی قرار دارند، نه روی یک ناحیه واضح داخل استخوان جمجمه. نام‌گذاری دقیق mimic نیازمند مرور متخصص است، اما از دید مدل این یک hard-negative acquisition/context است.

### تحلیل اندازه و اثر آن بر تصمیم 1024

در development، percentileهای 10/25/50/75/90 مساحت normalized باکس برابر `0.00313/0.00481/0.00839/0.01631/0.02950` است.

- 1839: median area=`0.00650`، تقریباً `50x31 px` در 512 و `75x46 px` در 768؛ boxهای منفرد percentile حدود 14 تا 60 development.
- 3356: median area=`0.00661`، تقریباً `48x37 px` در 512 و `71x56 px` در 768؛ percentile حدود 17 تا 62.
- 6871: median area=`0.09027` و تمام boxها بالاتر از percentile 90 development هستند.

در نتیجه 1839 و 3356 کوچک ولی extreme-small نیستند و 6871 بسیار بزرگ است. `imgsz=1024` هنوز ablation معقولی برای دو مورد اول است، اما audit فرض «مشکل صرفاً resolution است» را تأیید نمی‌کند و 6871 را توضیح نمی‌دهد.

### نتیجه و مسیر بعد

خطاها heterogeneous هستند. اولویت پیشنهادی:

1. hard-negative mining فقط از development/OOF برای نمونه‌های مشابه border/FOV/external equipment؛ test هرگز وارد training نشود.
2. آزمایش matched `768 vs 1024` روی OOF با seed/split ثابت؛ معیار تصمیم recall، PR-AUC و cross-fitted Binary QWK باشد، نه این چهار test case.
3. بررسی skull-region mask یا crop/context verifier برای حذف proposalهای خارج جمجمه. verifier به 3356/6871 کمک بالقوه می‌کند ولی 1839 را بدون high-recall proposal حل نمی‌کند.
4. ablation ورودی 2.5D در برابر single-channel یا context فاصله کمتر فقط روی OOF؛ overlayها اختلاف کانال sliceهای مجاور 5 mm را آشکار نشان می‌دهند و باید سود/زیان آن اندازه‌گیری شود.
5. calibration بعد از بهبود proposal recall انجام شود؛ calibration به‌تنهایی proposal failure را درمان نمی‌کند.

artifactهای کامل در `outputs/fixed_test_error_audit_v1/` قرار دارند و جمع‌بندی خوانا در `AUDIT_SUMMARY_FA.md` ذخیره شده است.

```bash
python -m json.tool outputs/fixed_test_error_audit_v1/audit_report.json
column -s, -t < outputs/fixed_test_error_audit_v1/study_model_summary.csv | less -S
find outputs/fixed_test_error_audit_v1/overlays -type f | sort | less
```

## طرح ablation پس از error audit — 2026-09-12

بر اساس بازخورد روش‌شناختی، اجرای هم‌زمان `1024 + HNM` به‌عنوان آزمایش اول رد شد، چون attribution اثر را ناممکن می‌کند. طرح مرحله‌ای فریز شد:

| Run | train/inference resolution | HNM | هدف |
|---|---:|---|---|
| A | 768 | No | baseline فعلی |
| B | 1024 | No | اثر خالص resolution بر proposal recall |
| C | 768 | Yes | اثر خالص hard negatives بر FP |
| D | 1024 | Yes | فقط در صورت مفیدبودن مستقل B یا C |

همه Runها باید patient-level OOF split یکسان، initialization یکسان از COCO، seed و سایر hyperparameterهای matched داشته باشند. در Run B، عبارت `1024-only` یعنی **هم آموزش و هم inference در 1024**؛ اجرای وزن 768 با `imgsz=1024` فقط یک diagnostic ارزان inference-scale است و جای ablation آموزشی B را نمی‌گیرد.

معیارهای فریز‌شده برای مقایسه A/B/C/D:

- GT proposal recall در سطح box با `conf>=0.001` و `IoU>=0.30`؛
- GT proposal recall با `IoU>=0.50`؛
- recall به‌صورت macro در سطح Study نیز، تا Studyهای دارای باکس زیاد غالب نشوند؛
- Study PR-AUC و ROC-AUC از continuous score؛
- cross-fitted Binary QWK، sensitivity، specificity، precision و FP count؛
- mAP50 و mAP50-95 به‌عنوان معیارهای تکمیلی localization، نه معیار انتخاب منفرد.

نکته ضد-leakage: Studyهای test یعنی `1839/3356/6871/271922` فقط منشأ فرضیه‌اند. حتی اگر بعداً خروجی 1024 برای `1839` proposal بسازد، این مشاهده نباید معیار انتخاب Run B باشد. انتخاب باید صرفاً از aggregate OOF/cross-fitted انجام شود و fixed test فعلی برای iterationهای بعدی مصرف‌شده تلقی گردد.

برای spatial filter نیز طراحی/تنظیم mask باید فقط روی development انجام شود. هدف اولیه حذف proposalهای بیرون skull/FOV است؛ hard-negative mining باید برای mimicهای باقی‌مانده، به‌ویژه موارد داخل skull، استفاده شود. هیچ crop یا proposal از Study `271922` نباید وارد train/HNM شود.

ترتیب تصمیم:

1. ابتدا baseline A باید با evaluator یکسان proposal-recall در تمام OOF ثبت شود.
2. سپس Run B به‌صورت 1024-only و matched اجرا شود.
3. فقط اگر deltaهای OOF از پیش تعریف‌شده مفید باشند، Run C برای spatial filtering/HNM به‌صورت جدا ساخته شود.
4. verifier و 2.5D ablation پس از روشن‌شدن bottleneckهای B/C بررسی شوند.

## ساخت evaluator و اندازه‌گیری authoritative Run A — 2026-09-12

### پیاده‌سازی و تعریف‌های فریز‌شده

`src/evaluate_oof_proposal_recall.py` ساخته شد تا یک evaluator مشترک برای A/B/C/D باشد. پنج مدل فقط روی held-out Fold متناظر inference می‌شوند. کنترل‌های hard-fail تضمین می‌کنند که فهرست‌ها تمام ۴۴۱۸ تصویر development را دقیقاً یک بار پوشش دهند، بیمار میان Foldها پخش نشود و هیچ مسیر fixed-test وارد evaluator نشود.

تنظیم Run A: `imgsz=768`، `batch=16`، `conf=0.001`، `NMS IoU=0.7` و `max_det=300`. recall اصلی coverage هر GT توسط حداقل یک proposal post-NMS است. size bin با مساحت pixel در تصویر اصلی تعریف شد تا با تغییر `imgsz` عوض نشود: small کمتر از `32²`، medium از `32²` تا کمتر از `96²` و large حداقل `96²`.

خروجی‌ها شامل raw match هر GT، prediction هر slice، Study score، cross-fitted decision، runtime هر Fold، peak CUDA allocated/reserved و quantileهای confidence هستند. سه تست برای binها، distribution و recall/missing count افزوده شد؛ کل suite **۲۵ تست پاس** است.

### خطاهای اجرایی و اصلاح

1. نخست reset حافظه CUDA در process تازه با خطای `Invalid device argument` متوقف شد. قبل از هر inference بود و خروجی ناقص نساخت. context با `torch.cuda.set_device/current_device` صریح initialize شد.
2. اجرای بعد پس از تمام inference به‌دلیل نام ناهماهنگ ستون `slice_proposals/proposals` در post-processing متوقف شد. تست مربوطه و نام ستون اصلاح شدند.
3. خروجی موفق نخست، VRAM را `null` ثبت کرد چون index GPU برابر صفر در شرط `if device` false ارزیابی شده بود. شرط به `device is not None` اصلاح و خروجی اولیه به‌صورت recoverable در `outputs/oof_proposal_recall_run_a_768_initial_no_vram` نگه داشته شد. اجرای authoritative از ابتدا تکرار و VRAM صحیح ثبت شد.

### نتایج proposal در کل OOF

۳0۷ GT box متعلق به ۲۴ Study مثبت ارزیابی شدند.

| اندازه | GT | no IoU30 | Recall IoU30 | no IoU50 | Recall IoU50 |
|---|---:|---:|---:|---:|---:|
| کل | 307 | 124 | 0.59609 | 142 | 0.53746 |
| Small | 51 | 21 | 0.58824 | 27 | 0.47059 |
| Medium | 227 | 83 | 0.63436 | 90 | 0.60352 |
| Large | 29 | 20 | 0.31034 | 25 | 0.13793 |

در ۷۸ GT، slice هیچ proposal post-NMS حتی در conf=0.001 نداشت. macro recall روی Studyهای مثبت برابر 0.57473 در IoU30 و 0.52745 در IoU50 است. نتیجه مهم این است که baseline فقط مشکل small object ندارد؛ large boxها بدترین recall را دارند. این می‌تواند ناشی از representation/annotation style/scale mismatch باشد و باید در Run B با همان evaluator بررسی شود.

برای matchهای IoU30، median بیشترین confidence منطبق 0.07483 و P10/P90 برابر 0.00238/0.56624 است. برای matchهای IoU50، median=0.08305 و P10/P90=`0.00244/0.56602` است. در largeها فقط چهار GT در IoU50 match شدند و median confidence=0.03446 بود. بنابراین low-confidence localization در کنار proposal failure گسترده وجود دارد.

### Study metrics و هزینه Run A

- PR-AUC=`0.55659`، ROC-AUC=`0.73204`.
- در threshold=0.5: sensitivity=`0.25` و QWK=`0.36386`.
- Cross-fitted: TP/FP/FN/TN=`8/7/16/138`، sensitivity=`0.3333` و Binary QWK=`0.33793`.
- زمان فراخوانی مدل برای ۴۴۱۸ تصویر=`58.76 s`؛ throughput تک‌مدل aggregate=`75.18 image/s`.
- کل evaluator با I/O و matching=`75.45 s`.
- max CUDA allocated=`1.948 GiB` و max reserved=`2.240 GiB` در batch 16.

خروجی authoritative در `outputs/oof_proposal_recall_run_a_768/` است و خلاصه خوانا در `BASELINE_SUMMARY_FA.md` قرار دارد. این مقادیر baseline فریز‌شده‌ی مقایسه Run B هستند؛ test در evaluator خوانده نشده است.

## Freeze و preflight آزمایش Run B — 2026-09-12

### Hypothesis و تنها متغیر اصلی

Run B اثر canvas/feature-map sampling در YOLO26s-P2 را می‌سنجد: train و inference از 768 به 1024 تغییر می‌کنند. هیچ اطلاعات فیزیکی جدیدی از CT ایجاد نمی‌شود. HNM، B25، spatial filter، verifier، calibration و aggregator جدید ممنوع‌اند. Fold membership، preprocessing، augmentation، optimizer، LR schedule، weight decay، patience و سقف epoch ثابت‌اند.

### بررسی rasterization

pipeline نوع A است: DICOM ابتدا با RescaleSlope/Intercept به HU تبدیل، با WL=800/WW=1600 window و با context فیزیکی ±5 mm به PNG سه‌کاناله **512x512** تبدیل شده است. نمونه‌های ابتدا/میانه/انتهای manifest همگی shape `(512,512,3)` داشتند. Run A فایل 512 را مستقیماً در YOLO به canvas 768 برده و Run B همان فایل 512 را مستقیماً به 1024 می‌برد. فایل raster ذخیره‌شده‌ی 768 و مسیر `DICOM 512 -> stored 768 -> YOLO 1024` وجود ندارد. بنابراین 1024 upscale است، نه افزایش resolution فیزیکی.

### فایل‌های جدید/تغییریافته

- `src/train_yolo26s_p2.py`: seed پیش از ساخت P2، init checkpoint اختصاصی هر run، protocol snapshot، train/val hash، runtime/VRAM و completion marker.
- `src/run_b_1024_kfold.py`: orchestrator اختصاصی و غیرقابل overwrite، waveهای حداکثر دو Fold، hard-fail بدون تغییر خودکار batch.
- پنج YAML در `data_prepared/skull_hu800_ww1600/run_b_1024/`: فقط train/val و بدون کلید test.
- `src/evaluate_oof_proposal_recall.py`: diagnostics هندسی ثانویه برای extent mismatch و hash وزن/list.
- `src/compare_run_a_b_1024.py`: pairing در سطح GT و Study، transitionها، patient-clustered bootstrap و verdict ازپیش‌تعریف‌شده.
- تست‌های `test_run_b_protocol.py` و `test_compare_run_a_b.py` و توسعه تست evaluator.

### Hashهای فریز‌شده

- COCO `weights/yolo26s.pt`: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`
- architecture: `2d57edc829bd8add780c6b60b043add292fd35373ed0f8b6f316287a4ba339e1`
- train Fold 0..4: `e8a5ec79...`, `dae3d685...`, `20b02f79...`, `5cb98ef0...`, `b3c8b31c...`
- val Fold 0..4: `25cb6e49...`, `040d9d15...`, `081f3511...`, `d9da0869...`, `1cf9cdca...`

hash کامل هر split در `outputs/run_b_1024_protocol.json` و سپس در `run_protocol.json` هر Fold ذخیره می‌شود.

### محدودیت matching initialization

Run A hash منبع COCO را ثبت کرده، اما hash checkpoint پیش از آموزشِ پارامترهای تصادفی P2 را برای هر Fold ذخیره نکرده و seed نیز قبل از model construction صریح set نشده بود. بنابراین random P2 initialization دقیق Run A قابل بازسازی نیست. Run B همان COCO source و seedهای Fold (`42..46`) را دارد و init hash اختصاصی را ذخیره می‌کند، اما این محدودیت به‌عنوان confound باقی می‌ماند و پنهان نمی‌شود.

### Diagnostic Large GT

برای بهترین-IoU proposal هر GT، نسبت area proposal/GT، intersection/GT، intersection/proposal، قرارگیری centerها، فاصله center نرمال‌شده با قطر GT، نسبت width/height، max confidence و proposal count افزوده شد. این diagnostics ثانویه‌اند و primary recall IoU30/50 را جایگزین نمی‌کنند. Run A با schema جدید بازتولید شد و primary metrics بدون تغییر ماند. در 29 Large GT فقط 11 مورد روی slice proposal داشتند؛ در همین 11 مورد median proposal/GT area=`0.5906`، median intersection/GT=`0.3913`، median intersection/proposal=`0.8718` و proposal center در 90.9% داخل GT بود. این شواهد annotation-extent mismatch/under-sized proposal را در بخشی از Largeها پشتیبانی می‌کند، در کنار 18 مورد proposal failure واقعی.

### تست و batch preflight

پیش از training، 30 تست نرم‌افزاری پاس و تمام فایل‌ها `py_compile` شدند. smoke با `imgsz=1024,batch=16,workers=0`, یک epoch و fraction=0.01 روی YAML بدون test کامل شد؛ OOM رخ نداد. peak CUDA allocated/reserved برابر `16.585/21.578 GiB` بود و completion marker ساخته شد. این smoke فقط تست نرم‌افزار/حافظه است و improvement مدل محسوب نمی‌شود. batch=16 برای Run B حفظ شد و هیچ کاهش خودکاری رخ نداده است.

فرمان smoke:

```bash
PYTHONPATH=src .venv/bin/python src/train_yolo26s_p2.py \
  --data data_prepared/skull_hu800_ww1600/run_b_1024/fold_0.yaml \
  --name run_b_1024_batch16_smoke --epochs 1 --imgsz 1024 \
  --batch 16 --workers 0 --device 0 --patience 30 --seed 42 --fraction 0.01
```

### شروع training اصلی، interruption و resume

اجرای اصلی با `batch=16` آغاز شد. در اجرای نخست Foldهای 0 و 1 هر کدام epoch 1 را کامل کردند، اما processهای وابسته به session اجرایی اولیه هنگام پایان آن session بدون traceback مدل و بدون پیام OOM خاتمه یافتند. فایل‌های `last.pt` هر دو Fold معتبر بودند. این رخداد یک interruption زیرساخت اجرا بود و نه نتیجه مدل یا خطای حافظه. یک تلاش `nohup` نیز به‌علت پاک‌سازی process پس‌زمینه توسط محیط orchestrator بلافاصله پایان یافت و log آن خالی بود.

برای ادامه قابل‌بازیابی، runner در session پایدار `tmux` اجرا و Foldهای 0 و 1 از `last.pt` مربوط به epoch 1 resume شدند. resume در `resume_events.json` ثبت می‌شود. batch تغییر نکرده است؛ در اجرای هم‌زمان دو Fold مصرف مشاهده‌شده حدود 18 GiB برای هر process بود و OOM رخ نداد.

فرمان authoritative runner:

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
tmux new-session -d -s yolo_run_b_1024 \
  "env PYTHONPATH=src .venv/bin/python src/run_b_1024_kfold.py --parallel 2 --resume-existing >> outputs/run_b_1024_runner.log 2>&1"
```

فرمان‌های مشاهده وضعیت بدون دخالت در training:

```bash
tmux attach -t yolo_run_b_1024
tail -f outputs/run_b_1024_logs/fold_0.log
tail -f outputs/run_b_1024_logs/fold_1.log
```

پس از تغییرات resume/evaluator/comparator، 8 تست هدفمند مجدداً پاس و چهار فایل اجرایی `py_compile` شدند. این فقط verification نرم‌افزار است و به‌عنوان improvement مدل تعبیر نمی‌شود.

یک audit مستقیم روی `args.yaml` Run A و Run B در Foldهای درحال‌اجرا انجام شد. در کل hyperparameterهای فریز‌شده، تنها اختلاف مؤثر `imgsz: 768 -> 1024` بود؛ مسیر `data` فقط از YAML اصلی به نسخه sanitized با membership یکسان تغییر کرده و مقدار `model` در فایل resume به مسیر `last.pt` همان run اشاره می‌کند. سپس کل suite اجرا شد و **30/30 تست پاس** شد. تلاش اجرای end-to-end comparator پیش از اتمام training نیز مطابق انتظار روی نبود `results.csv` فولد 2 متوقف شد؛ بنابراین comparator نمی‌تواند پیش از وجود artifactهای هر پنج Fold گزارش نهایی تولید کند.

### توقف پایش و handoff فرمان‌ها به کاربر

به درخواست کاربر، polling دوره‌ای training متوقف شد. آخرین بررسی یک‌باره نشان داد session مستقل `yolo_run_b_1024`، runner و processهای Foldهای 0 و 1 زنده‌اند. حلقه runner در waveهای `[0,1]`، سپس `[2,3]` و در پایان `[4]` اجرا می‌شود؛ بنابراین Foldهای بعدی پس از موفقیت و completion marker موج قبلی خودکار شروع می‌شوند. هیچ process restart نشد و هیچ config، batch یا hyperparameter تغییر نکرد. evaluator/comparator در این مرحله اجرا نشدند.

دستورهای exact مشاهده، کنترل completion و تحلیل پس از اتمام در پاسخ handoff ثبت شدند. fixed test نباید در هیچ‌یک از این فرمان‌ها وارد شود. Run C/D و verdict نیز تا دریافت نتایج اجرا نمی‌شوند.

### artifactها و دستور بازتولید

- `outputs/oof_fracture_threshold_v1/threshold_report.json`
- `outputs/oof_fracture_threshold_v1/oof_predictions.csv`
- `outputs/oof_fracture_threshold_v1/oof_crossfit_predictions.csv`
- `outputs/oof_fracture_threshold_v1/oof_threshold_sweep.csv`
- `outputs/oof_fracture_threshold_v1/fixed_test_secondary_predictions.csv`

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
source .venv/bin/activate
PYTHONPATH=src python src/select_oof_fracture_threshold.py
python -m json.tool outputs/oof_fracture_threshold_v1/threshold_report.json
column -s, -t < outputs/oof_fracture_threshold_v1/fixed_test_secondary_predictions.csv | less -S
```

## ممیزی مقایسه‌ای `SkullNet-main` با pipeline فعلی — 2026-09-12

### دامنه و محدودیت کار

این ممیزی فقط read-only روی کد، artifactهای OOF، metadata، annotationها و splitهای `SkullNet-main` انجام شد. training فعال Run B پایش، restart یا تغییر داده نشد؛ fixed test خوانده یا برای تصمیم‌گیری استفاده نشد؛ evaluator/comparator Run B نیز اجرا نشد.

### نتیجه اصلی: عدد QWK حدود 0.97 چه چیزی را اندازه می‌گیرد؟

فایل `reports/oof_official_spatial_v1/final_metrics.json` مقدار `isolated_fracture_qwk=0.9697234` را گزارش می‌کند، اما این **Binary QWK تشخیص شکستگی نیست**. تابع `src/fracture/evaluation/isolated_qwk.py` برای ICH و MLS از مقادیر ground-truth/oracle استفاده می‌کند و فقط fracture prediction را جایگزین می‌کند، سپس QWK سه‌کلاسه triage را می‌سنجد. بنابراین این عدد با Binary QWK pipeline ما قابل مقایسه مستقیم نیست.

از confusion matrix همان گزارش (`TP=12, TN=290, FP=20, FN=16`) Binary QWK واقعی برابر `0.3418434` و accuracy برابر `0.89349` است. در Run A ما Binary QWK در threshold ثابت 0.5 برابر `0.363864` و cross-fitted Binary QWK برابر `0.337932` است. cohortها متفاوت‌اند، پس حتی این اعداد هم head-to-head نیستند، ولی روشن می‌کنند که «0.97» شاهد detector تقریباً بی‌نقص نیست. در تحلیل اصلاح‌شده خود SkullNet، predictor همیشه منفی با صفر fracture detection روی همین isolated official metric حدود `0.988725` می‌گیرد؛ در نتیجه این metric به‌خاطر prevalence و oracle شدن دو primitive دیگر برای ارزیابی detector گمراه‌کننده است.

کد single-file پیوست‌شده پنج خروجی ICH و MLS را همگی صفر برمی‌گرداند. روی OOF موجود، شبیه‌سازی همین خروجی submission-like با score خام و threshold 0.5، QWK سه‌کلاسه حدود `0.04052` می‌دهد و اصولاً نمی‌تواند کلاس triage=2 تولید کند. این محاسبه diagnostic روی OOF است، نه ادعای leaderboard. اگر عدد 0.97 واقعاً از submission ترکیبی با مدل‌های مستقل ICH/MLS آمده، prediction/evaluator آن ترکیب برای نسبت دادن درست نتیجه لازم است.

همچنین فایل پیوست با `submit/model.py` فعلی یکسان نیست: نسخه فعلی threshold خام OOF برابر `0.1180267347` را با نگاشت monotonic به `fracture_prob=0.5` منتقل می‌کند؛ نسخه پیوست چنین rescaleای ندارد. `submit/best.pt` از نظر SHA256 دقیقاً checkpoint بهترین Fold 1 است (`17759933df8b783c6994baaddbb9758b848a2273e0280fc7c0adf0753922b0a8`). بنابراین artifact دقیق تولیدکننده QWK باید هنگام گزارش مشخص شود.

### ممیزی dataset و split

SkullNet کل dataset عمومی را به‌صورت زیر می‌بیند:

- 338 Study، 320 patient و 7683 DICOM؛ 28 Study مثبت.
- metadata معتبر برای 7508 slice: تعداد 260 مثبت و 7248 منفی صریح.
- 5176 JSON شامل 260 annotation مثبت و 4916 JSON خالی؛ 2332 slice دیگر JSON ندارند ولی در metadata صریحاً `SkullFracture=False` هستند.
- 175 DICOM فاقد metadata هستند و به‌درستی باید unknown بمانند و وارد loss نشوند.
- هیچ metadata/JSON بدون DICOM، duplicate `(series,SOP)`، mismatch مثبت بین JSON و metadata، یا Series با چند patient دیده نشد.

فایل `splits/folds.json` از نظر leakage صحیح است: همه 338 Study دقیقاً یک‌بار در validation پنج‌فولد پوشش داده می‌شوند؛ overlap بیمار و Study بین train/validation در همه Foldها صفر است؛ foldها 5 یا 6 Study مثبت دارند. 15 بیمار چند Study دارند ولی هیچ بیمار mixed-label وجود ندارد. بنابراین patient grouping فعلی درست است.

محدودیت روش‌شناختی SkullNet این است که held-out test داخلی ندارد و aggregator از میان چند گزینه روی کل OOF انتخاب و روی همان OOF گزارش شده است؛ calibration cross-fitted است اما انتخاب upstream aggregator کاملاً nested نیست. انتخاب Fold 1 به‌عنوان checkpoint نهایی نیز به‌علت تفاوت محسوس Foldها مستعد winner's curse است.

### مهم‌ترین تفاوتی که در pipeline خودمان باید اصلاح شود

`prepare_yolo26_dataset.py` فقط sliceهای دارای JSON را render می‌کند: 5176 slice از 198 Study. در development OOF فعلی ما 4418 تصویر از 169 Study، شامل 24 Study مثبت و 307 GT box وجود دارد. در نتیجه 2332 negative slice صریح metadata و تعداد زیادی Study منفی اصلاً وارد train نشده‌اند.

مهم‌تر از train، evaluator فعلی ما Study score را فقط روی فهرست تصاویر JSONدار validation می‌سازد؛ اما submission واقعی باید **تمام sliceهای CT** را inference و aggregate کند. proposal recall روی sliceهای GT همچنان معتبر است، ولی PR-AUC/QWK Study-level فعلی deployment-faithful نیست و ممکن است FPهای sliceهای بدون JSON را نبیند. SkullNet در `predict_fold.py` هر held-out Study را کامل از DICOM بارگذاری می‌کند. این مهم‌ترین نکته‌ای است که باید پیش از نتیجه‌گیری درباره architecture یا threshold اصلاح شود.

اصلاح پیشنهادی بدون تغییر Run B فریز‌شده:

1. proposal evaluator را روی GT sliceهای فعلی حفظ کنیم؛
2. یک evaluator مجزای full-study OOF بسازیم که برای هر بیمار held-out همه DICOMهای Study را score کند؛
3. در یک ablation مستقل بعدی، 2332 negative slice صریح metadata را وارد detector training کنیم و 175 unknown را mask/exclude نگه داریم؛
4. fixed internal test همچنان از development و هر تصمیم مدل جدا بماند. وزن‌های SkullNet که روی کل 338 Study OOF ساخته شده‌اند برای سنجش مستقل روی fixed test ما معتبر نیستند، چون آن cohort از همین داده عمومی منشأ گرفته است.

### preprocessing، training و postprocessing

پیش‌پردازش اصلی دو پروژه عملاً هم‌راستاست: HU با slope/intercept، fallback صحیح SimpleITK، MONOCHROME1، WL=800/WW=1600، مرتب‌سازی فیزیکی sliceها و context 2.5D در ±5 mm. در این بخش «راز» جدیدی که QWK 0.97 را توضیح دهد پیدا نشد. یک ریسک کدی برای هر دو pipeline این است که اگر یک پوشه چند `SeriesInstanceUID`/localizer داشته باشد، loader باید آن‌ها را صریح تفکیک یا reject کند؛ این یک ریسک robustness است و در این ممیزی به‌عنوان bug قطعی dataset اثبات نشد.

SkullNet v5 از YOLO11s-P2، imgsz=768، batch=2، AdamW با `lr0=5e-4`، `lrf=.05`، 120 epoch، patience=35، AMP خاموش، mosaic=0 و geometry ملایم‌تر استفاده می‌کند. train list آن negativeها را با ratio=3 downsample، positiveها و adjacent-negativeها را دوبرابر، و یک نسخه train-only با jitter پنجره HU می‌سازد. بنابراین بخش بزرگی از negativeهای معتبر در هر Fold یک‌بار کنار گذاشته شده‌اند. این strategy الزاماً بهتر از Run A/B ما نیست و نباید bundle کامل hyperparameterها کورکورانه کپی شود.

Postprocessing پیوست `conf=0.01`, NMS IoU=0.5 و `top3_mean` دارد؛ baseline proposal ما برای diagnostic از `conf=0.001`, NMS=0.7 استفاده می‌کند. این تفاوت‌ها باید فقط روی OOF development و به‌صورت ablation جدا سنجیده شوند. threshold انتخاب‌شده روی همان OOF apparent است؛ برای performance بی‌طرف باید cross-fitting/nested evaluation حفظ شود.

### اولویت آزمایش‌های بعد از اتمام Run B

1. **بدون retraining:** full-study patient-grouped OOF evaluation برای Run A و B؛ این اصلاح evaluator از همه فوری‌تر است.
2. **data-only ablation:** افزودن تمام negativeهای صریح metadata، با exclusion کامل unknownها.
3. مقایسه sampler پویا/patient-balanced یا all-negative training در برابر downsampling ثابت؛ تغییر فقط یک عامل در هر Run.
4. train-only HU-window jitter به‌عنوان ablation مستقل.
5. mosaic=0 و augmentation هندسی ملایم‌تر به‌صورت ablation مستقل، چون fracture باریک است و mosaic/scale قوی ممکن است signal را تخریب کند.
6. پس از فریز aggregator، calibration کاملاً cross-fitted. انتخاب single best Fold به‌جای refit/ensemble معتبر توصیه نمی‌شود.

هیچ‌یک از این موارد در این مرحله اجرا یا به Run B تزریق نشدند.

### آزمون نرم‌افزاری و provenance

- suite پروژه SkullNet در محیط موردنظر Python 3.12.4: `43/43 passed`.
- در Python 3.9 محیط YOLO_12: 37 pass و 6 fail، همگی به‌علت استفاده از `zip(..., strict=True)` که Python>=3.10 می‌خواهد. نسخه Python runtime مسابقه باید صریحاً کنترل شود.
- hash هر پنج `best.pt` ارزیابی‌شده با گزارش checkpoint مربوطه سازگار بود؛ `submit/best.pt` با Fold 1 یکسان است.
- هیچ model experiment یا inference جدید، fixed-test access یا تغییر فایل پروژه SkullNet انجام نشد. تنها فایل تغییرکرده در این ممیزی همین `agent.md` است.

### جمع‌بندی ممیزی

تقسیم patient-level SkullNet صحیح و بدون leakage آشکار است و استفاده از 2332 negative بدون JSON نیز به‌دلیل label منفی صریح metadata معتبر است. بااین‌حال QWK حدود 0.97 یک isolated triage metric با oracle ICH/MLS است، نه Binary fracture QWK و نه شاهد full-submission quality. ارزشمندترین انتقال از SkullNet به pipeline ما یک hyperparameter خاص نیست؛ **استفاده از کل Study در OOF inference و بهره‌گیری کنترل‌شده از تمام negativeهای صریح metadata** است.

## زیرساخت Macro-F1 و full-study OOF — 2026-09-12

### تغییر metric و دامنه کار

طبق اعلام جدید کاربر، primary metric رسمی از این مرحله **Macro-F1 سه‌کلاسه triage (0/1/2)** در نظر گرفته می‌شود و QWK فقط diagnostic است. این تغییر فقط در زیرساخت evaluation/post-processing جدید اعمال شد. هیچ فایل مسیر training/resume Run B، YAML، dataset آماده‌شده یا `model.py` تغییر نکرد. process فعال Run B poll/restart نشد، fixed-test result خوانده نشد و هیچ inference/training روی GPU اجرا نشد.

### فایل‌ها و رفتار پیاده‌سازی‌شده

- `src/triage_macro_f1.py`: port سخت‌گیرانه تابع رسمی `validate_intermediates` و `triage_from_intermediates` با threshold داخلی fracture برابر 0.5؛ گزارش pooled Macro-F1، precision/recall/F1/support هر کلاس، confusion matrix ثابت 3x3 و accuracy.
- `src/evaluate_official_triage_macro_f1.py`: evaluator عمومی CPU برای دو JSON با schema دقیق `study_id -> seven intermediates` و کنترل پوشش دقیق Studyها.
- `src/fracture_macro_f1_oof.py`: fracture-marginal OOF روی CSVهای held-out موجود؛ GT ICH/MLS به‌صورت oracle ثابت می‌ماند و metric فقط با نام `oracle_other_heads_macro_f1` گزارش می‌شود. baselineهای all-zero، all-one و raw OOF و نیز cross-fitted operating-point mapping را می‌سازد.
- `src/cache_full_study_oof.py`: scaffold قابل-resume برای inference همه sliceهای metadata-backed هر Study متعلق به بیمار held-out. unknownها قبل از inference حذف می‌شوند؛ cache هر Study شامل score/box/position هر slice است. overlap patient، اختلاف assignment، fixed-test inclusion و cache provenance mismatch باعث hard failure می‌شوند. Ultralytics به‌صورت lazy و فقط بعد از VRAM guard import می‌شود. CUDA با کمتر از 8 GiB حافظه آزاد، مگر با `--force` صریح، رد می‌شود.
- `src/aggregate_full_study_oof_cache.py`: aggregation کاملاً CPU از cache با candidate set فریز‌شده `max`, `top2_mean`, `top3_mean`, `top5_mean`, `top10_percent_mean`, `consecutive`. در هر held-out Fold، aggregator و operating point فقط روی چهار Fold دیگر انتخاب می‌شوند؛ primary selection برابر `oracle_other_heads_macro_f1` و guardrailها FP، specificity، sensitivity و PR-AUC هستند. raw operating point با نگاشت monotonic به probability-space منتقل می‌شود تا threshold رسمی پایین‌دست همیشه 0.5 بماند.
- `src/audit_oof_dataset_coverage.py`: audit سریع و CPU-only بر اساس metadata/pathها، بدون decode pixel؛ تعداد Study/patient/slice، JSON-backed، metadata-backed non-JSON، unknown، fold leakage، duplication/missing coverage و positive Study/patient هر Fold را گزارش می‌کند و dataset را تغییر نمی‌دهد.
- `tests/test_macro_f1_oof_infrastructure.py`: فقط synthetic data؛ قواعد مرزی triage، schema، confusion matrix، cross-fitting، monotonic mapping، candidate aggregators، VRAM guard و coverage audit را تست می‌کند.

### تعریف cross-fitting

برای Fold شماره `k`، operating point و در حالت cache-based خود aggregator فقط روی Foldهای غیر `k` انتخاب می‌شوند. score held-out با یک نگاشت piecewise-linear و monotonic تبدیل می‌شود که raw threshold انتخاب‌شده را دقیقاً روی `fracture_prob=0.5` قرار می‌دهد. سپس تابع رسمی بدون تغییر و با cutoff 0.5 اجرا می‌شود. threshold/نگاشت انتخاب‌شده روی همه OOF فقط deployment/apparent parameter است و unbiased performance نامیده نمی‌شود.

### تست‌های اجراشده

فقط تست سریع CPU و compile اجرا شد:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m py_compile \
  src/triage_macro_f1.py src/evaluate_official_triage_macro_f1.py \
  src/fracture_macro_f1_oof.py src/aggregate_full_study_oof_cache.py \
  src/cache_full_study_oof.py src/audit_oof_dataset_coverage.py \
  tests/test_macro_f1_oof_infrastructure.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_macro_f1_oof_infrastructure.py
```

نتیجه: **9/9 passed in 0.82s**؛ `py_compile` نیز موفق بود. این‌ها software verification هستند و هیچ ادعایی درباره improvement مدل ایجاد نمی‌کنند.

### فرمان‌های CPU قابل اجرا در زمان training

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/fracture_macro_f1_oof.py \
  --output outputs/fracture_macro_f1_run_a_oof

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/audit_oof_dataset_coverage.py \
  --output outputs/oof_dataset_coverage_audit.json
```

فرمان دوم صرفاً metadata و نام pathها را می‌خواند و pixel DICOM را decode نمی‌کند. هیچ‌یک Ultralytics/YOLO را import نمی‌کنند.

### فرمان full-study OOF بعد از پایان کامل Run B

این فرمان در این مرحله اجرا نشده و فقط پس از پایان هر پنج Fold Run B مجاز است:

```bash
PYTHONPATH=src .venv/bin/python src/cache_full_study_oof.py \
  --score-protocol deployment_score \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold0/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold1/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold2/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold3/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold4/weights/best.pt \
  --cache-dir outputs/full_study_oof_run_b_1024_deployment_score_cache \
  --device 0 --imgsz 1024 --batch 16 --half
```

پس از completion cache، aggregation بدون GPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/aggregate_full_study_oof_cache.py \
  --cache-dir outputs/full_study_oof_run_b_1024_deployment_score_cache \
  --output outputs/full_study_oof_run_b_1024_macro_f1
```

### وضعیت آزمایش

در این مرحله فقط کد و تست synthetic کامل شده است. Run A Macro-F1 واقعی، coverage audit واقعی، full-study OOF، انتخاب aggregator از cache و هرگونه verdict مدل **اجرا نشده‌اند**. Run C/D/HNM شروع نشده و `model.py` بدون تغییر باقی مانده است.

## اصلاح full-study inclusion و ممیزی 175 DICOM unknown — 2026-09-12

### ورودی authoritative و سیاست unknown

CPU coverage audit کاربر passed تلقی شد: 338 Study، 320 patient، 7683 DICOM، 7508 slice metadata-backed، 5176 JSON-backed، 2332 metadata-backed non-JSON و 175 metadata-unknown. development شامل 169 Study/155 patient است و پنج Fold بدون patient leakage، duplication یا missing coverage آن را کامل پوشش می‌دهند؛ 28 بیمار fixed-test جدا هستند.

نبود metadata/annotation فقط slice را از detector supervision خارج می‌کند و مجوز ساخت label منفی یا مثبت نمی‌دهد؛ اما حذف خودکار از deployment inference نیز درست نیست. scaffold اصلاح شد تا DICOM بدون metadata، در صورت تعلق قطعی به target CT Series و سازگاری هندسی، وارد full-study inference شود.

### تغییرات پیاده‌سازی

- `src/dicom_series_guard.py`: header-only discovery، unique SOP guard، canonical `SeriesInstanceUID`، تشخیص scout/localizer، Rows/Columns و orientation guard و ordering از projection موقعیت روی normal تصویر. fallback فقط وقتی مجاز است که physical position برای همه sliceها غایب ولی `InstanceNumber` همه موجود و unique باشد؛ mixed availability hard-fail است.
- `src/audit_metadata_unknown_dicoms.py`: ممیزی تمام unknownها با `stop_before_pixels=True` و خروجی CSV/JSON.
- `src/cache_full_study_oof.py`: همان guard مشترک را قبل از pixel decode اعمال می‌کند؛ unknown معتبر را include و هر foreign/scout/incompatible را با path/classification/reason در cache ثبت می‌کند. provenance، resume، patient/fixed-test hard-fail و VRAM guard حفظ شدند.
- `src/compare_full_study_oof_caches.py`: comparator CPU-only با completion/coverage/protocol identity guards، cross-fitted aggregator و operating point مستقل برای A/B، delta در جهت B-A و paired stratified patient-clustered bootstrap.
- `src/aggregate_full_study_oof_cache.py`: protocol identity را validate می‌کند. بالا بردن confidence floor در cache با NMS ثابت ممکن است و تست شده، ولی `nms_reconstructed=false` ثبت می‌شود؛ NMS متفاوت هرگز از post-NMS cache بازسازی نمی‌شود.
- `src/triage_macro_f1.py`: Micro-F1 نیز گزارش می‌شود که در single-label multiclass برابر accuracy است.
- تست‌های synthetic برای header classification/ordering، protocolها، confidence filtering و paired comparator افزوده شدند.

### score-generation protocolهای فریز‌شده

```text
proposal_diagnostic: conf=0.001, NMS IoU=0.7, max_det=300
purpose: proposal recall/localization only

deployment_score: conf=0.01, NMS IoU=0.5, max_det=300
initial Study aggregator: top3_mean
purpose: Macro-F1, calibration and final aggregator selection
```

نام/پارامتر protocol، `study_inclusion_protocol=target_series_header_guard_v1_all_valid_dicoms`، imgsz و weight SHA256 در cache و completion marker ذخیره می‌شوند. comparator فقط A/B با یک score protocol و inclusion logic یکسان را قبول می‌کند؛ تفاوت مجاز اصلی weight و `768->1024` است.

### نتیجه exact ممیزی headerهای unknown

فرمان اجراشده:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/audit_metadata_unknown_dicoms.py
```

7683 header در 7.5 ثانیه بررسی شد؛ pixel array decode نشد:

```text
metadata_unknown_dicoms: 175
studies_with_unknown_dicoms: 22
include_target_series: 175
exclude_foreign_series: 0
exclude_localizer_or_scout: 0
exclude_incompatible_geometry: 0
unresolved: 0
header_read_errors: 0
SeriesInstanceUID matches canonical: 175/175
geometry/orientation compatible: 175/175
Modality CT: 175/175
Rows/Columns 512x512: 175/175
ImageType ORIGINAL/PRIMARY/AXIAL: 175/175
ImagePositionPatient/ImageOrientationPatient/PixelSpacing/SliceThickness present: 175/175
InstanceNumber absent: 18/175؛ ordering همچنان فیزیکی و معتبر است.
```

پس هر 175 مورد برای deployment/full-study inference شامل می‌شوند ولی برای training هیچ label دریافت نمی‌کنند. artifactها:

- `outputs/metadata_unknown_dicom_audit.csv`
- `outputs/metadata_unknown_dicom_audit_summary.json`

### comparator و محدودیت CI

برای Fold k، aggregator و raw operating point فقط با Foldهای `!=k` انتخاب می‌شوند؛ boundary خام با نگاشت monotonic به `fracture_prob=0.5` منتقل و Fold k فقط evaluate می‌شود. خروجی pooled شامل oracle-other-heads Macro-F1، Micro-F1/accuracy، classwise metrics/support، confusion matrix 3x3، fracture TP/FP/FN/TN، sensitivity/specificity/precision، PR-AUC، ROC-AUC و Binary QWK diagnostic است.

Bootstrap روی predictionهای cross-fitted فریز‌شده، paired و stratified patient-clustered است. این CI sampling uncertainty بیمار را پوشش می‌دهد اما selection procedure را داخل هر bootstrap دوباره fit نمی‌کند و تمام selection uncertainty را مدل نمی‌کند.

### verification

```text
py_compile: passed
synthetic CPU unit tests: 13/13 passed in 0.88s
GPU/Ultralytics/pixel inference: not executed
Run B poll/restart/change: none
fixed-test results accessed: none
```

### فرمان‌های آینده؛ فقط پس از اتمام Run B

Run A با `deployment_score` و imgsz=768:

```bash
PYTHONPATH=src .venv/bin/python src/cache_full_study_oof.py \
  --score-protocol deployment_score \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold0/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold1/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold2/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold3/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold4/weights/best.pt \
  --cache-dir outputs/full_study_oof_run_a_768_deployment_score_cache \
  --device 0 --imgsz 768 --batch 16 --half
```

Run B با همان protocol و imgsz=1024:

```bash
PYTHONPATH=src .venv/bin/python src/cache_full_study_oof.py \
  --score-protocol deployment_score \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold0/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold1/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold2/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold3/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold4/weights/best.pt \
  --cache-dir outputs/full_study_oof_run_b_1024_deployment_score_cache \
  --device 0 --imgsz 1024 --batch 16 --half
```

مقایسه paired پس از completion هر دو cache:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
  src/compare_full_study_oof_caches.py \
  --cache-a outputs/full_study_oof_run_a_768_deployment_score_cache \
  --cache-b outputs/full_study_oof_run_b_1024_deployment_score_cache \
  --output outputs/full_study_oof_ab_deployment_score_comparison \
  --bootstrap-repeats 2000 --seed 20260912
```

هیچ Run C/D/HNM، تغییر augmentation، افزودن training negative یا تغییر `model.py` انجام نشد.

## ممیزی provenance cohort و اصلاح cache به v2_169 — 2026-09-13

### فرضیه و دلیل ممیزی

دو cache کامل Run A/B هر کدام 172 Study داشتند، درحالی‌که predictionهای OOF قدیمی روی 169 Study صریح validation ساخته شده بودند. شمار 4418 مربوط به image/slice pathهای validation است و با Study count مقایسه نشد. فرضیه بررسی‌شده این بود که cacher نسخه اول از patient held-out به تمام Studyهای metadata آن patient گسترش یافته است، نه اینکه فقط sliceهای Study صریح validation را گسترش دهد.

این ممیزی CPU-only بود. comparator اجرا نشد، Ultralytics/CUDA/pixel inference اجرا نشد، fixed-test result خوانده نشد و cacheهای 172تایی حذف یا overwrite نشدند.

### cohort authoritative بازیابی‌شده از manifest

تابع جدید هیچ Study ID را از filename حدس نمی‌زند. هر path در `fold_*_val.txt` با ستون `image_path` resolved در `manifest.csv` به Study/patient نگاشت می‌شود؛ نگاشت path باید one-to-one باشد. نتیجه دقیق:

```text
Fold 0: 900 validation slices, 33 Studies, 31 patients
1488, 1525, 1938, 1939, 1952, 2200, 2686, 270803, 270845, 271003, 271013,
271159, 271313, 271422, 271528, 271823, 271935, 272179, 272190, 272229, 272443,
272520, 272597, 272626, 272665, 3347, 3604, 4066, 4445, 689, 7478, 835, 9248

Fold 1: 877 validation slices, 33 Studies, 31 patients
1164, 1232, 2068, 2069, 2265, 2336, 2386, 2388, 2638, 270967, 271067, 271208,
271221, 271361, 271497, 271590, 271623, 271686, 272022, 272114, 272355, 2998,
3059, 3082, 3206, 3416, 3796, 519, 5658, 663, 904, 925, 9255

Fold 2: 869 validation slices, 32 Studies, 31 patients
1544, 1843, 1923, 1975, 2301, 2441, 270806, 270907, 271335, 271508, 271635,
271843, 271869, 271874, 272519, 272624, 272645, 272689, 3088, 3172, 3405,
342682, 4276, 515, 554, 6709, 704, 7317, 7428, 902, 989, 990

Fold 3: 885 validation slices, 36 Studies, 31 patients
1011, 1030, 1062, 1080, 1934, 1996, 1997, 1999, 2034, 2424, 270957, 271016,
271061, 271569, 271762, 272173, 272589, 3125, 342691, 3608, 3905, 4167, 4240,
4310, 4325, 4326, 4327, 447, 674, 6845, 6846, 7045, 7053, 7203, 880, 883

Fold 4: 887 validation slices, 35 Studies, 31 patients
1032, 1033, 1560, 1956, 2024, 2063, 2558, 270818, 270859, 270960, 271032,
271033, 271663, 271792, 271912, 2722, 272205, 272468, 272528, 2732, 2847, 2850,
2947, 2951, 2952, 2953, 342675, 465, 598, 613, 673, 7150, 7396, 917, 9244

Total unique explicit validation Studies: 169
Per-fold invariant: [33, 33, 32, 36, 35]
```

هیچ Study تکراری، Study گم‌شده یا patient مشترک بین Foldهای validation وجود نداشت.

### مقایسه محتوای دو cache قدیمی

Study ID و patient ID از داخل JSON هر cache استخراج شد، نه از تعداد entryهای filesystem. A و B نتیجه یکسان داشتند:

```text
Fold 0: expected=33, cached=33, extra=[], missing=[]
Fold 1: expected=33, cached=33, extra=[], missing=[]
Fold 2: expected=32, cached=32, extra=[], missing=[]
Fold 3: expected=36, cached=37, extra=[3122], missing=[]
Fold 4: expected=35, cached=37, extra=[2023, 2557], missing=[]
```

discrepancy provenance:

```text
Fold 3: Study 3122, patient 770803؛ Study صریح همان patient در Fold 3 = 3125
Fold 4: Study 2023, patient 700862؛ Study صریح همان patient در Fold 4 = 2024
Fold 4: Study 2557, patient 767996؛ Study صریح همان patient در Fold 4 = 2558
```

Artifact ممیزی کامل، شامل expected/cached exact IDs هر Fold برای هر دو cache:

- `outputs/full_study_oof_cohort_provenance_audit.json`

### علت ریشه‌ای تأییدشده

نسخه قبلی `cache_full_study_oof.py` پس از استخراج patientهای held-out این کار را انجام می‌داد:

```python
fold_metadata = metadata[metadata["patient_id"].isin(patients)]
```

در نتیجه تمام Studyهای metadata متعلق به آن patient وارد cache می‌شدند. این رفتار در سه patient چند-Study باعث 172 به‌جای 169 شد. patient leakage رخ نداده بود، ولی **Study cohort drift** رخ داده بود. full-study expansion باید فقط sliceهای درون Study صریح validation را گسترش دهد، نه تعداد Studyها را.

### اصلاح پیاده‌سازی و invariants

- `src/oof_cohort.py` اضافه شد: cohort صریح را فقط با manifest mapping استخراج می‌کند و invariants `total=169` و `[33,33,32,36,35]` را hard-enforce می‌کند.
- `src/audit_full_study_cache_cohort.py` اضافه شد: expected cohort را با Study IDs داخل JSON cache مقایسه و discrepancy patientها را ثبت می‌کند.
- `src/cache_full_study_oof.py` اصلاح شد: انتخاب اکنون `explicit Study IDs -> metadata/headerهای همان Study -> تمام DICOMهای معتبر target Series` است. patient فقط guard است و دیگر cohort selector نیست.
- `src/compare_full_study_oof_caches.py` اصلاح شد: پیش از هر مقایسه cohort را دوباره از manifest/val lists می‌سازد و completion marker و JSON content هر Fold باید دقیقاً expected باشند. extra یا missing hard-fail می‌شود.
- protocol جدید: `manifest_mapped_explicit_validation_studies_v2_169` و `explicit_validation_studies_v2_all_valid_target_series_dicoms`. بنابراین cache قدیمی به‌اشتباه با v2 resume نمی‌شود.
- `tests/test_macro_f1_oof_infrastructure.py` توسعه یافت: patient دارای دو Study، حفظ همه sliceهای Study منتخب، پوشش صریح exact، extra-cache failure و missing-cache failure.

قبل از import Ultralytics و GPU inference، total/per-fold cohort، unique Study، patient-fold exclusivity، fixed-test exclusion و هر cache موجود بررسی می‌شود. در resume، missing Study مجاز است فقط تا وقتی Fold completion marker ندارد؛ extra Study همیشه hard-fail است. cache کامل با missing Study hard-fail می‌شود.

از 175 metadata-unknown DICOM معتبر، دقیقاً 24 slice در چهار Study cohort صحیح 169تایی قرار دارند و در inference v2 وارد می‌شوند:

```text
1952: 8
2200: 6
2388: 7
2722: 3
```

unknownها همچنان هیچ training label دریافت نمی‌کنند.

### تست و اجرای واقعی

فرمان‌های اجراشده:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/audit_full_study_cache_cohort.py \
  --cache outputs/full_study_oof_run_a_768_deployment_score_cache \
  --cache outputs/full_study_oof_run_b_1024_deployment_score_cache \
  --output outputs/full_study_oof_cohort_provenance_audit.json

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m py_compile \
  src/oof_cohort.py src/audit_full_study_cache_cohort.py src/cache_full_study_oof.py \
  src/compare_full_study_oof_caches.py tests/test_macro_f1_oof_infrastructure.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_macro_f1_oof_infrastructure.py
```

نتیجه: **17/17 passed in 0.97s** و `py_compile` موفق. این software verification و provenance audit است، نه model improvement. هیچ A/B comparator، GPU inference یا retraining اجرا نشد.

### cacheهای قدیمی و فرمان‌های صحیح v2

دو cache 172تایی زیر حفظ شده‌اند و فقط diagnostic هستند؛ comparator اصلاح‌شده آن‌ها را به‌دلیل extra Study رد می‌کند:

- `outputs/full_study_oof_run_a_768_deployment_score_cache`
- `outputs/full_study_oof_run_b_1024_deployment_score_cache`

Run A v2، فقط اجرای دستی پس از آزاد شدن GPU:

```bash
PYTHONPATH=src .venv/bin/python src/cache_full_study_oof.py \
  --score-protocol deployment_score \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold0/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold1/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold2/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold3/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_fold4/weights/best.pt \
  --cache-dir outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169 \
  --device 0 --imgsz 768 --batch 16 --half
```

Run B v2، همان protocol و cohort:

```bash
PYTHONPATH=src .venv/bin/python src/cache_full_study_oof.py \
  --score-protocol deployment_score \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold0/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold1/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold2/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold3/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_run_b_1024_fold4/weights/best.pt \
  --cache-dir outputs/full_study_oof_run_b_1024_deployment_score_cache_v2_169 \
  --device 0 --imgsz 1024 --batch 16 --half
```

Comparator فقط پس از completion هر دو v2 cache:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
  src/compare_full_study_oof_caches.py \
  --cache-a outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169 \
  --cache-b outputs/full_study_oof_run_b_1024_deployment_score_cache_v2_169 \
  --output outputs/full_study_oof_ab_deployment_score_comparison_v2_169 \
  --bootstrap-repeats 2000 --seed 20260912
```

قدم بعدی فقط اجرای دستی دو cache v2 توسط کاربر و سپس comparator است. Run C/D/HNM شروع نشده و fixed test وارد این ممیزی نشده است.

## materialization بدون inference و مقایسه نهایی Run A/B روی cohort دقیق 169تایی — 2026-09-13

### فرضیه و قید تصمیم

فرضیه Run B این بود که با ثابت ماندن protocol، افزایش canvas شبکه از 768 به 1024 ممکن است representation و در نتیجه معیار اصلی triage را بهتر کند. معیار اصلی از قبل `oracle_other_heads_macro_f1` تعیین شده بود؛ این معیار ICH volume و MLS واقعی را به‌عنوان oracle وارد می‌کند و **Macro-F1 نهایی submission نیست**. Binary QWK فقط diagnostic است. fixed test در materialization، انتخاب cross-fitted یا مقایسه خوانده نشد؛ GPU inference/retraining نیز اجرا نشد.

### ساخت cacheهای authoritative بدون محاسبه مجدد YOLO

فایل جدید `src/filter_full_study_cache_cohort.py` cohort صریح validation را از manifest mapping می‌سازد، Study identity را از محتوای JSON cache می‌خواند، وجود دقیق سه extra تأییدشده و نبود missing را hard-check می‌کند و فقط 169 Study مورد انتظار را به destination جدید materialize می‌کند. برای جلوگیری از تغییر payload، فایل‌های Study مستقیماً hard-link شدند و markerهای `COMPLETE.json` و `FILTERED_CACHE_PROVENANCE.json` جدید و مستقل ساخته شدند.

نتیجه واقعی برای هر دو Run:

```text
total Studies = 169
per Fold = [33, 33, 32, 36, 35]
hard links = 169
fallback copies = 0
prediction recomputation = false
excluded Fold 3 = [3122]
excluded Fold 4 = [2023, 2557]
```

cacheهای 172تایی منبع حذف یا overwrite نشدند. تمام payloadهای Study منتخب byte-identical باقی ماندند؛ بنابراین تمام per-slice predictionها، از جمله metadata-unknownهای معتبر موجود در Studyهای منتخب، حفظ شدند.

### دروازه provenance پیش از comparator

فایل جدید `src/validate_filtered_full_study_caches.py` قبل از comparator هر دو cache را مستقل بررسی کرد. artifact نتیجه:

- `outputs/full_study_oof_filtered_cache_v2_169_validation.json`

نتیجه `PASS` بود:

```text
A = 169, B = 169
per Fold = [33, 33, 32, 36, 35]
A Study/patient/fold identities == B identities
missing = []
extra = []
duplicate = 0
fixed-test Study overlap = []
fixed-test patient overlap = []
cross-Fold patient leakage = false
score protocol = deployment_score
conf = 0.01
NMS IoU = 0.5
A imgsz = 768
B imgsz = 1024
protocol equal beyond fold-specific weights and imgsz = true
per-Study target/metadata-unknown slice counts identical between A/B = true
```

### فرمان‌های دقیق اجراشده

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/filter_full_study_cache_cohort.py \
  --source outputs/full_study_oof_run_a_768_deployment_score_cache \
  --destination outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169 \
  --expected-imgsz 768

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/filter_full_study_cache_cohort.py \
  --source outputs/full_study_oof_run_b_1024_deployment_score_cache \
  --destination outputs/full_study_oof_run_b_1024_deployment_score_cache_v2_169 \
  --expected-imgsz 1024

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/validate_filtered_full_study_caches.py \
  --cache-a outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169 \
  --cache-b outputs/full_study_oof_run_b_1024_deployment_score_cache_v2_169 \
  --output outputs/full_study_oof_filtered_cache_v2_169_validation.json

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
  src/compare_full_study_oof_caches.py \
  --cache-a outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169 \
  --cache-b outputs/full_study_oof_run_b_1024_deployment_score_cache_v2_169 \
  --output outputs/full_study_oof_ab_deployment_score_comparison_v2_169 \
  --bootstrap-repeats 2000 --seed 20260912
```

Comparator فقط cacheهای held-out OOF را استفاده کرد. برای هر held-out Fold، aggregator و operating point با چهار Fold دیگر انتخاب شد؛ سپس نگاشت monotonic باعث شد operating point خام به threshold رسمی `fracture_prob=0.5` نگاشت شود.

### نتیجه paired cross-fitted OOF

| معیار | Run A، 768 | Run B، 1024 | Delta B-A |
|---|---:|---:|---:|
| `oracle_other_heads_macro_f1` | 0.969627 | 0.954958 | -0.014670 |
| Micro-F1 / accuracy | 0.970414 | 0.952663 | -0.017751 |
| Class 0 F1 | 0.977778 | 0.977778 | 0.000000 |
| Class 1 F1 | 0.952381 | 0.924528 | -0.027853 |
| Class 2 F1 | 0.978723 | 0.962567 | -0.016157 |
| Fracture TP / FP / FN / TN | 7 / 3 / 17 / 142 | 4 / 6 / 20 / 139 | TP -3، FP +3، FN +3، TN -3 |
| Sensitivity | 0.291667 | 0.166667 | -0.125000 |
| Specificity | 0.979310 | 0.958621 | -0.020690 |
| Precision | 0.700000 | 0.400000 | -0.300000 |
| Study PR-AUC | 0.535549 | 0.405493 | -0.130056 |
| ROC-AUC | 0.729885 | 0.785489 | +0.055603 |
| Binary QWK diagnostic | 0.358147 | 0.165591 | -0.192556 |

Triage confusion matrix، سطر=true و ستون=predicted، labels `[0,1,2]`:

```text
Run A                  Run B
[[22, 1, 0],           [[22, 1, 0],
 [ 0,50, 1],            [ 0,49, 2],
 [ 0, 3,92]]            [ 0, 5,90]]
```

Classwise precision/recall/F1:

```text
Run A: class0=1.000000/0.956522/0.977778
       class1=0.925926/0.980392/0.952381
       class2=0.989247/0.968421/0.978723
Run B: class0=1.000000/0.956522/0.977778
       class1=0.890909/0.960784/0.924528
       class2=0.978261/0.947368/0.962567
```

Cross-fitted selections به ترتیب Fold 0 تا 4:

```text
Run A:
0 top5_mean            0.172503662109375
1 top10_percent_mean   0.39717610677083337
2 top10_percent_mean   0.41237386067708337
3 top10_percent_mean   0.39717610677083337
4 max                  0.443603515625

Run B:
0 top10_percent_mean   0.3231658935546875
1 top5_mean            0.739697265625
2 top5_mean            0.739697265625
3 top10_percent_mean   0.21162923177083331
4 top5_mean            0.806591796875
```

Patient-clustered stratified paired bootstrap با 2000 تکرار و seed=`20260912` برای Delta Macro-F1 برابر بود با:

```text
B-A = -0.0146697366
95% percentile CI = [-0.0519539668, 0.0196879241]
```

CI صفر را قطع می‌کند؛ بنابراین این sample آسیب قطعی آماری 1024 را اثبات نمی‌کند. بااین‌حال، معیار پذیرش از پیش فریز شده «شاهد OOF برای بهترشدن» بود و Run B آن را برآورده نکرد: معیار اصلی پایین‌تر است، PR-AUC و sensitivity/precision/specificity همگی پایین‌ترند و FP و FN هر دو سه مورد بیشترند. تنها ROC-AUC بهتر شده و برای جبران شکست معیار اصلی/guardrailها کافی نیست.

proposal-recall paired برای Run B artifact موجود ندارد؛ فقط گزارش Run A موجود است. بنابراین هیچ proposal metric نابرابر یا inference جدیدی وارد این verdict نشد و برای تکمیل آن GPU inference اجرا نشد.

Artifactهای نتیجه:

- `outputs/full_study_oof_ab_deployment_score_comparison_v2_169/report.json`
- `outputs/full_study_oof_ab_deployment_score_comparison_v2_169/paired_cross_fitted_predictions.csv`

### تست نرم‌افزاری و محدودیت

تست synthetic جدید ثابت می‌کند extra Study حذف، همه sliceهای Study منتخب حفظ، payload byte-identical و completion coverage دقیق است. نتیجه نهایی:

```text
18 passed in 0.92s
py_compile: PASS
```

این تست‌ها شاهد بهبود مدل نیستند. نتیجه OOF با 24 Study مثبت fracture و oracle بودن دو head دیگر محدود است و CI نیز عدم قطعیت را نشان می‌دهد. این گزارش نه fixed-test است و نه Macro-F1 واقعی full submission.

### verdict OOF-only

`REJECT_1024`

دلیل: 1024 هیچ بهبود OOF در معیار اصلی ایجاد نکرد و هم‌زمان guardrailهای عملی fracture را بدتر کرد؛ بنابراین دلیلی برای حفظ هزینه بیشتر 1024 در pipeline فعلی وجود ندارد. این verdict به معنی اثبات قطعی آماری مضر بودن 1024 نیست، بلکه رد آن طبق acceptance rule فریز‌شده این ablation است. هیچ Run بعدی، HNM، verifier یا تغییر `model.py` شروع نشد.

## تعریف و پیاده‌سازی AUG_MILD_768 — 2026-09-13

### فرضیه و محدوده

نام رسمی این ablation برابر `AUG_MILD_768` است و عمداً Run C نیست؛ نام تاریخی Run C برای `768 + HNM` رزرو شده است. فرضیه این است که mosaic و geometry نسبتاً قوی Run A ممکن است morphology باریک شکستگی را تخریب کند و augmentation ملایم‌تر شاید evidence/recall را با FP محدود بهتر کند. این فرضیه هنوز آزمایش نشده است. در این نوبت هیچ training، GPU inference، fixed-test evaluation، HNM، B25، verifier، spatial filter یا تغییر dataset انجام نشد.

### پروتکل واقعی بازیابی‌شده Run A

منابع بررسی‌شده: `args.yaml` و `pretrained_transfer_audit.json` هر پنج Fold، logهای Run A، `src/train_yolo26s_p2.py`، config معماری و train/val lists. مقادیر واقعی:

```yaml
model: YOLO26s-P2, Detect heads P2/P3/P4/P5
architecture: configs/yolo26s-p2.yaml
architecture_sha256: 2d57edc829bd8add780c6b60b043add292fd35373ed0f8b6f316287a4ba339e1
coco_source: weights/yolo26s.pt, official Ultralytics YOLO26s COCO
coco_sha256: 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
input: HU WL=800/WW=1600, 2.5D physical context +/-5 mm
imgsz: 768
batch: 16
epochs: 150
patience: 30
workers: 0
optimizer: AdamW
lr0: 0.001
lrf: 0.01
adam_betas: [0.937, 0.999]
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
cos_lr: true
amp: true
seed_by_fold: [42, 43, 44, 45, 46]
deterministic: true
fraction: 1.0
nbs: 64
cache: false
rect: false
multi_scale: 0.0
save_period: 10
box: 7.5
cls: 0.5
dfl: 1.5
mosaic: 0.30
degrees: 5.0
translate: 0.05
scale: 0.15
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
close_mosaic: 10
```

`auto_augment` و `erasing` در args ثبت شده‌اند، هرچند مسیر detection ممکن است آن‌ها را مانند classification مصرف نکند؛ برای جلوگیری از drift صریحاً freeze شدند. تنها diff مجاز و تست‌شده:

```text
mosaic:    0.30 -> 0.00
degrees:   5.00 -> 3.00
translate: 0.05 -> 0.03
scale:     0.15 -> 0.08
```

تمام مقادیر دیگر بالا ثابت‌اند. snapshot کامل machine-readable در `outputs/aug_mild_768_protocol.json` ذخیره شد. runner پیش از شروع، SHA256 معماری/COCO و هر ده train/val list را با مقادیر Run A hard-check می‌کند. تعداد تصاویر train/val Foldهای 0 تا 4 به‌ترتیب `3518/900`، `3541/877`، `3549/869`، `3533/885` و `3531/887` است. configهای جدید همان listها را بدون کلید test ارجاع می‌دهند؛ prepared dataset تغییر نکرد.

### confound initialization

Run A تاریخی random P2 initialization قابل بازسازی ندارد، زیرا seed-before-construction و checkpoint اولیه هر Fold به‌طور قابل اثبات ثبت نشده بود. AUG_MILD_768 قبل از ساخت مدل seed می‌کند، `coco_init.pt` هر Fold را نگه می‌دارد و SHA256 آن و COCO source را در `run_protocol.json` ثبت می‌کند. بنابراین مقایسه آینده با Run A دارای initialization confound است. matched five-fold control اکنون آموزش داده نشد.

### فایل‌ها و guards

- `src/aug_mild_768_protocol.py`: پروتکل، exact diff، architecture/COCO/split guards، resume/completion validation.
- `src/train_aug_mild_768_fold.py`: seeded initialization، ذخیره/hash checkpoint اولیه، train/resume یک Fold.
- `src/run_aug_mild_768_kfold.py`: اجرای ترتیبی `0->1->2->3->4`، skip Fold کامل و resume فقط Fold ناقص.
- `configs/aug_mild_768_data/fold_0.yaml` تا `fold_4.yaml`: همان membership و بدون test.
- `tests/test_aug_mild_768_protocol.py`: diff، hashes، forbidden modes، output isolation، resume drift، completion و cohort 169.
- `outputs/aug_mild_768_protocol.json`: snapshot فریز‌شده قبل از training.

output prefix مستقل `outputs/yolo26s_p2_hu800_ww1600_aug_mild_768_fold{0..4}` است و نمی‌تواند Run A/B را overwrite کند. balanced sampling/HNM/extra-negative mode وجود ندارد. auto batch reduction وجود ندارد. resume، augmentation داخل protocol و checkpoint را بررسی می‌کند. completion معتبر نیازمند `training_completed.json`، `weights/best.pt`، `results.csv` و `run_protocol.json` معتبر است.

### تست‌های CPU/static

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m py_compile \
  src/aug_mild_768_protocol.py src/train_aug_mild_768_fold.py \
  src/run_aug_mild_768_kfold.py tests/test_aug_mild_768_protocol.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_aug_mild_768_protocol.py tests/test_macro_f1_oof_infrastructure.py \
  tests/test_run_b_protocol.py
```

نتیجه نهایی پس از افزودن تست مستقل checkpoint-resume: `27 passed in 3.45s` و `py_compile: PASS`. `Ultralytics get_cfg` نیز تمام overrideهای صریح را پذیرفت. این software validation است، نه model improvement. هیچ training output directory ساخته نشد.

### فرمان دستی training و resume

شروع ترتیبی در tmux:

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
tmux new-session -d -s aug_mild_768 \
  'cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12 && exec env PYTHONPATH=src .venv/bin/python -u src/run_aug_mild_768_kfold.py --device 0 > outputs/aug_mild_768_runner.log 2>&1'
```

فقط پس از قطع واقعی runner، ادامه امن:

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
tmux new-session -d -s aug_mild_768 \
  'cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12 && exec env PYTHONPATH=src .venv/bin/python -u src/run_aug_mild_768_kfold.py --device 0 --resume-existing >> outputs/aug_mild_768_runner.log 2>&1'
```

status دستی و بدون loop:

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
tmux has-session -t aug_mild_768 2>/dev/null && echo 'tmux: RUNNING' || echo 'tmux: NOT RUNNING'
for FOLD in 0 1 2 3 4; do
  RUN="outputs/yolo26s_p2_hu800_ww1600_aug_mild_768_fold${FOLD}"
  if [ -f "${RUN}/training_completed.json" ] && [ -f "${RUN}/weights/best.pt" ] && [ -f "${RUN}/results.csv" ] && [ -f "${RUN}/run_protocol.json" ]; then
    echo "Fold ${FOLD}: COMPLETE"
  elif [ -f "${RUN}/weights/last.pt" ]; then
    echo "Fold ${FOLD}: INCOMPLETE/RESUMABLE"
  else
    echo "Fold ${FOLD}: NOT STARTED"
  fi
done
tail -n 20 outputs/aug_mild_768_runner.log
nvidia-smi
```

### فرمان آینده full-study OOF cache، فقط بعد از پایان پنج Fold

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
PYTHONPATH=src .venv/bin/python src/cache_full_study_oof.py \
  --score-protocol deployment_score \
  --weights outputs/yolo26s_p2_hu800_ww1600_aug_mild_768_fold0/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_aug_mild_768_fold1/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_aug_mild_768_fold2/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_aug_mild_768_fold3/weights/best.pt \
  --weights outputs/yolo26s_p2_hu800_ww1600_aug_mild_768_fold4/weights/best.pt \
  --cache-dir outputs/full_study_oof_aug_mild_768_deployment_score_cache_169 \
  --device 0 --imgsz 768 --batch 16 --half
```

این evaluator cohort صریح `[33,33,32,36,35]`، تمام DICOMهای معتبر target-Series و metadata-unknownهای معتبر داخل Studyهای منتخب را hard-enforce می‌کند. protocol برابر `deployment_score` با conf=0.01 و NMS IoU=0.5 است. فرمان اجرا نشد.

### فرمان آینده comparator با Run A authoritative

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
  src/compare_full_study_oof_caches.py \
  --cache-a outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169 \
  --cache-b outputs/full_study_oof_aug_mild_768_deployment_score_cache_169 \
  --expected-a-imgsz 768 --expected-b-imgsz 768 \
  --output outputs/full_study_oof_run_a_vs_aug_mild_768_comparison_169 \
  --bootstrap-repeats 2000 --seed 20260912
```

Comparator همان exact 169 Studies، cross-fitted aggregator/operating point، نگاشت monotonic به cutoff رسمی 0.5، oracle-other-head Macro-F1 و paired patient-clustered bootstrap را به کار می‌برد. معیار اصلی `oracle_other_heads_macro_f1` است؛ TP/FN و محدود ماندن FP guardrail هستند. تا پایان training/cache هیچ نتیجه دقتی ادعا نمی‌شود.

## ساخت artifact سابمیشن YOLO26s-P2 بر اساس Macro-F1 — 2026-09-13

### محدوده شواهد و انتخاب checkpoint

در زمان انتخاب، Run B 1024 با verdict برابر `REJECT_1024` کنار گذاشته شده بود. AUG_MILD_768 فقط پوشه Fold 0 را داشت، فاقد `training_completed.json` و فاقد OOF پنج‌فولدی/cache بود؛ بنابراین وارد انتخاب نشد. انتخاب فقط از پنج checkpoint کامل Run A 768 و predictionهای authoritative full-study OOF روی 169 Study انجام شد؛ fixed test خوانده نشد.

Macro-F1 held-out هر checkpoint با aggregation/operating point cross-fitted همان Fold:

```text
Fold 0: 0.9285714286   top5_mean           threshold=0.172503662109375
Fold 1: 0.9764982374   top10_percent_mean  threshold=0.39717610677083337
Fold 2: 1.0000000000   top10_percent_mean  threshold=0.41237386067708337
Fold 3: 0.9789147287   top10_percent_mean  threshold=0.39717610677083337
Fold 4: 0.9658994032   max                 threshold=0.443603515625
```

بر اساس معیار درخواستی، checkpoint منتخب Fold 2 است:

```text
source: outputs/yolo26s_p2_hu800_ww1600_fold2/weights/best.pt
packaged: submit/models/best.pt
SHA256: 6c3635082f4ed0e15d549e02e3a7faeaec0293299a06b01d342d8f2e3853f206
```

دو فایل وزن byte-identical هستند. انتخاب checkpoint از foldهای دارای patient cohortهای متفاوت و تنها 32 Study held-out در Fold 2 انجام شده است؛ بنابراین عدم قطعیت انتخاب زیاد است و Macro-F1 برابر 1.0 نباید به‌عنوان برآورد population یا leaderboard گزارش شود.

### post-processing سابمیشن

روی تمام 169 OOF Study، apparent/deployment selection برابر بود با:

```text
aggregator = top10_percent_mean
raw operating point = 0.39717610677083337
official fracture cutoff after monotonic mapping = 0.5
apparent all-OOF oracle_other_heads_macro_f1 = 0.9851285969
```

این مقدار apparent است، چون operating point روی تمام OOF انتخاب و روی همان OOF گزارش شده است. تخمین محافظه‌کارانه pooled cross-fitted برای Run A همان `oracle_other_heads_macro_f1=0.9696273781` با TP/FP/FN/TN شکستگی `7/3/17/142` است.

نکته حیاتی: هر دو عدد از **ground-truth ICH volumes و ground-truth MLS** استفاده می‌کنند. `submit/model.py` فعلی fracture-only است و پنج ICH volume و MLS را صفر می‌دهد؛ بنابراین `0.969627` یا `0.985129` Macro-F1 نهایی همین artifact مستقل نیست. برای submission کامل تیم باید headهای واقعی ICH/MLS با این fracture output ادغام شوند.

### محتویات artifact

```text
submit/model.py
submit/models/best.pt
submit/SELECTION_REPORT.json
submit/README.md
```

`model.py` single-file و offline است و موارد زیر را داخل خود دارد: DICOM discovery/ordering، decode با pydicom و fallback SimpleITK، تبدیل HU، WL=800/WW=1600، ورودی 2.5D فیزیکی ±5 mm، YOLO26s-P2 در imgsz=768، conf=0.01، NMS IoU=0.5، max_det=300، top10-percent-mean، mapping monotonic و API رسمی `Model().predict(study_dir)`. وزن از `submit/models/best.pt` resolve می‌شود. `YOLO_OFFLINE=true` و مسیر config قابل‌نوشتن `/tmp/Ultralytics` پیش از import Ultralytics تنظیم می‌شود.

در مقایسه با فایل قدیمی SkullNet، threshold خام `0.1180267` و `top3_mean` منتقل نشدند، زیرا model-version-specific بودند. مقادیر جدید از OOF همین YOLO26 Run A به دست آمده‌اند. Batch inference برابر 16 است؛ این با cache authoritative مطابقت دارد و داخل محدودیت 24GB باید جداگانه روی محیط رسمی benchmark شود.

### تست‌های انجام‌شده

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m py_compile \
  submit/model.py tests/test_submit_package.py

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_submit_package.py
```

نتیجه: `4 passed in 0.21s`. تست‌ها hash وزن، ثابت‌های protocol، top10-percent aggregation، mapping operating point به 0.5، تطبیق byte-for-byte preprocessing synthetic با implementation پروژه و schema دقیق هفت‌کلیدی را بررسی کردند.

بارگذاری واقعی offline/CPU موفق بود:

```text
weights=/mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12/submit/models/best.pt
aggregation=top10_percent_mean
raw_threshold=0.39717610677083337
model_type=YOLO
```

دو inference end-to-end روی development held-out Fold 2 انجام شد، نه fixed test:

```text
Study 1843: 16 slices, raw=0.0, seven-key output PASS
Study 3088: 16 slices
cached GPU/FP16 raw=0.785888671875
packaged CPU/FP32 raw=0.7861654758453369
absolute difference=0.00027680397033691406
mapped fracture_prob=0.82263930929379
```

اختلاف کوچک raw در Study 3088 ناشی از backend/precision متفاوت CPU-FP32 و GPU-FP16 است. artifact از نظر syntax، checkpoint load، preprocessing، aggregation و API واقعاً اجرا شد. این تست عملکرد hidden یا Macro-F1 نهایی را اثبات نمی‌کند.

تست نهایی از داخل خود پوشه `submit` با import رسمی `from model import Model` نیز پاس شد و وزن را از `submit/models/best.pt` resolve کرد. `submit/__pycache__` تولیدشده توسط این تست از artifact نهایی خارج و به پوشه موقت قابل‌بازیابی `/tmp/yolo12-submit-pycache.RW3AJm` منتقل شد. محتویات نهایی فقط `model.py`، `models/best.pt`، `README.md` و `SELECTION_REPORT.json` است.



# ============================================================
# MODEL ROUTING AND SUBAGENT POLICY
# ============================================================

The main agent is responsible for final decisions, implementation,
and verification.

The goal of subagent routing is to preserve high reasoning quality while
avoiding unnecessary use of the expensive main model for bulk reading.


## Mandatory read-heavy delegation

For substantial repository exploration, DO NOT perform broad read-heavy
work directly in the main agent.

You MUST delegate the broad discovery work to the `explorer` agent when
the task requires any of the following:

- searching across multiple files
- locating implementations across the repository
- running multiple find, rg, grep, git grep, or similar discovery commands
- reading more than 3 source files to understand an area of the codebase
- traversing or listing multiple directories
- inspecting large logs
- inspecting large CSV or JSON experiment outputs
- collecting results from multiple experiment folders
- tracing call sites across multiple files
- mapping a pipeline or repository architecture
- inspecting many tests to understand expected behavior

The explorer should perform the bulk reading/searching and return a concise,
evidence-backed summary to the main agent.

The explorer should report:

- exact file paths
- relevant functions/classes/symbols
- important values or observations
- relevant line ranges when useful
- uncertainty or missing evidence

The main agent may directly read a small number of critical files or sections
after explorer identifies them, in order to verify important conclusions.


## When direct main-agent reading is acceptable

The main agent may read directly when:

- only one or two already-known files are needed
- checking a small configuration file
- checking a small diff
- verifying one critical code section
- inspecting a file that explorer already identified as decisive
- the overhead of spawning explorer would exceed the work itself

Do not spawn explorer for trivial single-file work.


## Deep-analysis routing

Use the `analyst` agent when the problem genuinely requires unusually deep
reasoning, including:

- difficult root-cause debugging
- several plausible competing hypotheses
- architecture redesign
- subtle algorithmic correctness
- machine-learning methodology
- patient/study-level data leakage
- cross-validation correctness
- train/validation/test contamination
- metric implementation correctness
- threshold-selection bias
- calibration correctness
- aggregation correctness
- complex interactions across multiple components

The analyst should receive distilled evidence from explorer whenever possible,
instead of performing broad repository exploration itself.


## Do not use analyst for

Do NOT invoke analyst for:

- routine repository navigation
- find/grep/rg searches
- ordinary file reading
- simple questions
- formatting
- straightforward coding
- mechanical edits
- simple bugs
- extracting values from logs
- collecting experiment metrics


## Preferred workflow

For repository-heavy difficult problems, prefer:

1. Main agent determines what evidence is required.
2. Explorer performs broad read-heavy investigation.
3. Explorer returns concise evidence.
4. Main agent reviews the evidence.
5. If normal reasoning is sufficient, main agent solves the problem.
6. If unusually deep reasoning is required, delegate the distilled problem
   and evidence to analyst.
7. Analyst performs deep reasoning.
8. Main agent verifies the important evidence.
9. Main agent makes the final decision and implements changes.


## Efficiency rules

Do not spawn subagents for trivial tasks.

Prefer one appropriate subagent over several redundant agents.

Do not run multiple agents on the same task unless independent verification
has meaningful value.

Do not return entire large logs, CSVs, JSON files, or source files to the
main agent.

Return only the evidence required for reasoning.

Do not repeatedly reread unchanged files.

For large files, locate the relevant section first and read only the required
range.


## Quality rule

Efficiency must never override correctness.

If explorer results are incomplete, ambiguous, contradictory, or critical
to a high-impact decision, obtain additional evidence before concluding.

The main agent must not blindly accept a cheaper agent's conclusion.

Explorer gathers evidence.
Analyst performs exceptional deep reasoning.
The main agent owns the final technical conclusion, implementation,
and verification.
## ارزیابی پایان آموزش AUG_MILD_768 و مقایسه با Run A/Run B — 2026-09-15

### هدف و روش

درخواست کاربر این بود که پس از پایان آخرین آموزش، ارزیابی بر مبنای Macro-F1 انجام و با آموزش‌ها و تست قبلی مقایسه شود. معیار اصلی این بخش `oracle_other_heads_macro_f1` است: پیش‌بینی fracture از مدل OOF می‌آید، ولی حجم ICH و MLS از ground truth گرفته می‌شوند و triage با تابع رسمی سه‌کلاسه ساخته می‌شود. بنابراین این عدد **Macro-F1 نهایی submission نیست** و تنها سهم حاشیه‌ای head شکستگی را در triage اندازه می‌گیرد. Binary QWK فقط diagnostic است.

هیچ inference تازه‌ای روی fixed test انجام نشد. fixed test قبلاً مصرف شده و گزارش موجود آن فقط fracture-only binary است؛ در نتیجه با Macro-F1 سه‌کلاسه OOF مقایسه مستقیم نمی‌شود. انتخاب مدل فقط از patient-grouped OOF انجام شد.

کوهورت authoritative در هر سه اجرا دقیقاً ۱۶۹ Study و ۱۵۵ patient است؛ تعداد Study در Foldها `[33, 33, 32, 36, 35]` است. مقایسه‌گر، برابری Study/patient/fold، توالی مرتب SOPها و ۲۴ slice با metadata نامشخص را برای تمام ۴۴۴۲ slice بین cacheها اثبات کرد. تنظیم score هر سه cache `deployment_score` با `conf=0.01` و NMS IoU=0.5 است.

### تغییر نرم‌افزاری لازم برای مقایسه

- `src/compare_full_study_oof_caches.py`: دو برچسب قدیم/جدید study-inclusion فقط زمانی معادل پذیرفته می‌شوند که exact Study/patient/fold identity، ordered SOP sequence و metadata-unknown count برای تک‌تک Studyها برابر باشند. اختلاف واقعی slice همچنان hard-fail است.
- `tests/test_macro_f1_oof_infrastructure.py`: تست synthetic برای پذیرش اختلاف صرفاً اسمی protocol در صورت پوشش SOP کاملاً یکسان، و تست hard-fail برای SOP متفاوت اضافه شد.

### فرمان‌های دقیق اجراشده

Aggregation آخرین آموزش از cache کامل‌شده، CPU-only:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
  src/aggregate_full_study_oof_cache.py \
  --cache-root outputs/full_study_oof_aug_mild_768_deployment_score_cache_169 \
  --output outputs/full_study_oof_aug_mild_768_deployment_score_aggregate_169
```

مقایسه paired با Run A:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
  src/compare_full_study_oof_caches.py \
  --cache-a outputs/full_study_oof_run_a_768_deployment_score_cache_v2_169 \
  --cache-b outputs/full_study_oof_aug_mild_768_deployment_score_cache_169 \
  --expected-a-imgsz 768 --expected-b-imgsz 768 \
  --output outputs/full_study_oof_run_a_vs_aug_mild_768_comparison_169_v2 \
  --bootstrap-repeats 2000 --seed 20260915
```

مقایسه paired با Run B 1024:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
  src/compare_full_study_oof_caches.py \
  --cache-a outputs/full_study_oof_run_b_1024_deployment_score_cache_v2_169 \
  --cache-b outputs/full_study_oof_aug_mild_768_deployment_score_cache_169 \
  --expected-a-imgsz 1024 --expected-b-imgsz 768 \
  --output outputs/full_study_oof_run_b_1024_vs_aug_mild_768_comparison_169 \
  --bootstrap-repeats 2000 --seed 20260915
```

### نتیجه واقعی OOF

| معیار | Run A 768 | Run B 1024 | AUG_MILD 768 جدید |
|---|---:|---:|---:|
| oracle-other-heads Macro-F1 | **0.969627** | 0.954958 | 0.938469 |
| Micro-F1 / Accuracy | **0.970414** | 0.952663 | 0.946746 |
| F1 کلاس 0 | **0.977778** | **0.977778** | 0.930233 |
| F1 کلاس 1 | **0.952381** | 0.924528 | 0.917431 |
| F1 کلاس 2 | **0.978723** | 0.962567 | 0.967742 |
| fracture TP / FP / FN / TN | **7 / 3 / 17 / 142** | 4 / 6 / 20 / 139 | 3 / 14 / 21 / 131 |
| fracture sensitivity | **0.291667** | 0.166667 | 0.125000 |
| fracture specificity | **0.979310** | 0.958621 | 0.903448 |
| fracture precision | **0.700000** | 0.400000 | 0.176471 |
| Study PR-AUC | **0.535549** | 0.405493 | 0.334950 |
| ROC-AUC | 0.729885 | **0.785489** | 0.745115 |

Confusion matrix سه‌کلاسه با ترتیب کلاس‌های `[0,1,2]`:

- Run A: `[[22,1,0],[0,50,1],[0,3,92]]`
- Run B: `[[22,1,0],[0,49,2],[0,5,90]]`
- AUG_MILD: `[[20,3,0],[0,50,1],[0,5,90]]`

مقایسه paired آموزش جدید با Run A:

- delta Macro-F1 جدید منهای A: `-0.0311588160`
- patient-clustered bootstrap 95% CI: `[-0.0802712260, 0.0111044470]`
- delta Study PR-AUC: `-0.2005989025`؛ 95% CI=`[-0.3691519159,-0.0263236106]`
- FP برابر ۱۱ مورد بیشتر و FN برابر ۴ مورد بیشتر شد.

مقایسه paired آموزش جدید با Run B:

- delta Macro-F1 جدید منهای B: `-0.0164890794`
- patient-clustered bootstrap 95% CI: `[-0.0627474680,0.0271716295]`
- FP برابر ۸ مورد بیشتر و FN برابر ۱ مورد بیشتر شد.

### proposal diagnostic

در AUG_MILD، proposal recall کلی فقط اندکی بهتر شد: IoU30 از `0.596091` به `0.599349` و IoU50 از `0.537459` به `0.540717`. تعداد GT روی slice بدون هیچ proposal همچنان ۷۸ بود. این تغییر کوچک با افت مهم Study PR-AUC و افزایش FP همراه است. IoU50 برای small از `0.470588` به `0.392157` و large از `0.137931` به `0.103448` افت کرد؛ medium از `0.603524` به `0.629956` بهتر شد. بنابراین افزایش بسیار کوچک recall کلی دلیل کافی برای پذیرش مدل جدید نیست.

### تست نرم‌افزاری و خطا

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_macro_f1_oof_infrastructure.py tests/test_aug_mild_768_protocol.py
```

نتیجه: **27 passed** در 3.96 ثانیه. این نتیجه صحت مسیر نرم‌افزاری را می‌سنجد و شاهد افزایش دقت مدل نیست.

اولین فرمان `py_compile` به‌دلیل واردکردن نام اشتباه و ناموجود `src/aggregate_full_study_oof.py` شکست خورد؛ فایل درست `src/aggregate_full_study_oof_cache.py` است. پس از اصلاح فرمان، compile زیر بدون خطا تمام شد:

```bash
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/compare_full_study_oof_caches.py \
  src/aggregate_full_study_oof_cache.py \
  src/triage_macro_f1.py
```

### مقایسه با fixed test قبلی و محدودیت

تنها fixed-test موجود مربوط به ensemble پنج Fold Run A و fracture-only است: ۲۹ Study با فقط ۴ Study مثبت. در threshold خام 0.5 نتیجه `TP=0, FP=0, FN=4, TN=25`، sensitivity=0، specificity=1، PR-AUC=`0.397645` و Binary QWK=0 بود. بازتحلیل post-hoc با threshold مشتق‌شده از OOF نتیجه `TP=1, FP=1, FN=3, TN=24` و Binary QWK=`0.265823` داشت. این‌ها Macro-F1 سه‌کلاسه نیستند و fixed test قبلاً دیده شده بود؛ بنابراین برای انتخاب AUG_MILD استفاده نشد و inference تازه روی آن اجرا نشد.

### تصمیم

**بهترین آموزش فعلی بر اساس full-study patient-grouped OOF و معیار اصلی Macro-F1، Run A 768 است.** AUG_MILD رد می‌شود، چون هم Macro-F1، هم PR-AUC، هم sensitivity، specificity و precision را نسبت به Run A کاهش داده و FP را از ۳ به ۱۴ رسانده است. Run A باید checkpoint ensemble مرجع باقی بماند. عدد `0.969627` یک `oracle_other_heads_macro_f1` توسعه‌ای است و نباید به‌عنوان Macro-F1 نهایی/hidden-test submission گزارش شود.

## ادامه پردازش پکیج submission پس از انتخاب نهایی — 2026-09-15

پس از تکرار درخواست کاربر برای ادامه پردازش، پکیج موجود `submit/` بررسی شد. این پکیج از قبل checkpoint مربوط به Run A Fold 2 را نگه می‌داشت. SHA256 فایل بسته‌بندی‌شده با منبع اصلی دقیقاً برابر بود:

```text
6c3635082f4ed0e15d549e02e3a7faeaec0293299a06b01d342d8f2e3853f206
```

وزن یا `submit/model.py` تغییر نکرد. `submit/SELECTION_REPORT.json` که هنوز AUG_MILD را ناقص معرفی می‌کرد با نتیجه نهایی پنج‌Fold اصلاح شد؛ Run B 1024 با Macro-F1=`0.9549576415` و AUG_MILD با Macro-F1=`0.9384685621` به‌عنوان candidate ردشده ثبت شدند. `submit/README.md` نیز با مقایسه authoritative سه اجرا و تصریح عدم استفاده از fixed test برای انتخاب به‌روز شد.

فایل‌های تغییرکرده در این مرحله:

- `submit/SELECTION_REPORT.json`
- `submit/README.md`
- `agent.md`

تست CPU-only اجراشده:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_submit_package.py \
  tests/test_macro_f1_oof_infrastructure.py \
  tests/test_aug_mild_768_protocol.py
```

نتیجه: **31 passed in 4.11s**. سپس JSON گزارش parse شد، checkpoint واقعی با Ultralytics روی CPU load شد و SHA256 دوباره بررسی شد؛ نتیجه `cpu_checkpoint_load=passed` بود. `py_compile submit/model.py` نیز بدون خطا گذشت. این‌ها تست سازگاری نرم‌افزاری/بسته‌بندی هستند و معیار جدید مدل محسوب نمی‌شوند.

محدودیت پابرجاست: artifact فعلی fracture-only است و پنج حجم ICH و MLS را صفر برمی‌گرداند؛ بنابراین `0.969627` عدد oracle-other-head OOF برای انتخاب detector است، نه Macro-F1 نهایی همین artifact مستقل. ادغام headهای واقعی ICH/MLS برای submission کامل همچنان لازم است.

## آماده‌سازی refit نهایی Run A روی full169 — 2026-09-15

هدف، refit واحد خانواده منتخب Run A 768 روی کل development بود؛ هیچ training، inference یا دسترسی به نتایج fixed test انجام نشد. cohort از union پنج `fold_*_val.txt` ساخته و هر path فقط از طریق `manifest.csv` به Study/patient نگاشت شد. نتیجه دقیق: 169 Study، 155 patient و 4418 تصویر با تعداد Fold برابر `[33,33,32,36,35]` و تصویر برابر `[900,877,869,885,887]`. overlap Study/patient با ردیف‌های manifest دارای split=test صفر، duplicate صفر و Tier-2/metadata-only افزوده‌شده صفر است. union دقیقاً با همه تصاویر development موجود Run A برای این 169 Study برابر است. SHA256 cohort برابر `3acb49a60be94a3e0750d308e1fb92848ca9a8a672421437adaedc8f68fd9203` است.

best epochهای تاریخی Run A از `results.csv` برابر `[42,99,61,20,59]`، mean=`56.2` و median=`59` هستند. طبق قاعده از پیش خواسته‌شده، fixed epochs نهایی 59 انتخاب شد.

فایل‌های جدید: `src/run_a_full169_protocol.py`، `src/train_run_a_full169.py`، `tests/test_run_a_full169_protocol.py` و پوشه `configs/run_a_full169/` شامل `full169_train.txt`، `syntax_only_val.txt`، `dataset.yaml` و `cohort_report.json`. هیچ output تاریخی یا submission تغییر نکرد.

Runner seed=42 را پیش از model construction اعمال می‌کند؛ architecture/COCO/init/cohort/config hash و snapshot کامل را ذخیره می‌کند؛ batch=16 ثابت است و هر تلاش Ultralytics برای کاهش خودکار batch hard-fail می‌شود. `val=False` است. چون YAML در Ultralytics الزاماً `val` می‌خواهد، یک تصویر داخل همان training cohort فقط به‌عنوان syntax-only val معرفی شده است؛ اگر Ultralytics در finalization آن را بسنجد، هیچ metricی برای انتخاب checkpoint استفاده نمی‌شود. deployment checkpoint نسخه byte-identical از final-epoch `weights/last.pt` با نام `weights/final_deployment.pt` است؛ `best.pt` مبنای deployment نیست.

`py_compile` سه فایل جدید پاس شد. تست CPU-only برابر `13 passed in 8.20s` بود و status خروجی `not_started` داد. بررسی صریح نیز تأیید کرد مسیر `outputs/yolo26s_p2_hu800_ww1600_run_a_full169` هنوز وجود ندارد. آموزش شروع نشده است.

## آماده‌سازی FITNESS_BALANCED_768 — 2026-09-15

یک آزمایش isolated پنج‌Fold با membership، seedهای 42 تا 46 و تمام training/augmentationهای Run A آماده شد. تنها متغیر آزمایش checkpoint selector ثانویه است:

```text
balanced_fitness = 0.10*P + 0.30*R + 0.10*mAP50 + 0.50*mAP50-95
```

وزن‌ها جمعاً 1.0 هستند. `BalancedCheckpointTrainer` هیچ‌یک از `save_model`، `validate` یا `_do_train` را override نمی‌کند؛ در نتیجه fitness پیش‌فرض `[0,0,0,1]`، `best.pt` و early stopping عادی Ultralytics بدون تغییر باقی می‌مانند. callback محلی پس از ذخیره checkpoint هر validation epoch، فقط در صورت افزایش strict امتیاز، `last.pt` همان epoch را به‌صورت atomic در `weights/best_balanced.pt` کپی می‌کند و epoch، P/R/mAPها، fitness و SHA256 را در `weights/best_balanced.json` ثبت می‌کند. رویداد callback اضافی final-eval با guard رد می‌شود. پس از پایان training، checkpoint balanced strip و hash metadata به‌روز خواهد شد.

configهای sanitized جدید هیچ کلید test ندارند و hash train/val آنها با Run A دقیقاً برابر است. output prefix با Run A، Run B، AUG_MILD و FULL169 تداخل ندارد. هیچ فایل زیر `.venv/site-packages` تغییر نکرد. فایل‌ها: `src/fitness_balanced_768_protocol.py`، `src/balanced_fitness_checkpoint.py`، `src/train_fitness_balanced_768_fold.py`، `src/run_fitness_balanced_768_kfold.py`، `tests/test_fitness_balanced_768.py` و پنج YAML در `configs/fitness_balanced_768_data/`.

`py_compile` پاس شد. تست‌های CPU/static برابر `14 passed in 3.41s` بود و strict improvement، فرمول، جمع وزن، inherited بودن normal selection/early stopping، split hashes و output isolation را پوشش داد. status هر پنج Fold `not_started` است و هیچ output آزمایش ساخته نشده است. FULL169 لمس/متوقف نشد؛ training و inference اجرا نشد.

## آماده‌سازی MOSAIC08_SCALE05_BALANCED_FOLD4 — 2026-09-15

این continuation تک‌Fold و post-hoc از checkpoint تاریخی Run A Fold 4 با SHA256=`30e953e6f60b85b8e940559eb24c69b8817d46d27fcc3479b1fab0807c75ac16` آماده شد. منبع فقط هنگام شروع دستی به `initialization_source.pt` در output جدید copy و از همان copy با `resume=False` load می‌شود؛ در نتیجه optimizer/schedule تازه‌اند. hash منبع پیش و پس از آماده‌سازی یکسان ماند و output جدید هنوز وجود ندارد.

تنها تغییر augmentation نسبت به Run A عبارت است از mosaic `0.30->0.80` و scale `0.15->0.50`؛ degrees=5 و translate=0.05 ثابت‌اند. max epochs=40 انتخاب شد، چون checkpoint از قبل آموزش‌دیده است، best epoch تاریخی Fold 4 برابر 59 بود و convention بهتر دیگری برای continuation در repo یافت نشد. patience=30 و best.pt/early stopping پیش‌فرض Ultralytics حفظ شدند. selector ثانویه همان فرمول `0.10P+0.30R+0.10mAP50+0.50mAP50-95` را به `best_balanced.pt` و metadata ریشه run با نام `best_balanced_metrics.json` می‌نویسد. کاهش خودکار batch از 16 hard-fail است.

فایل‌های جدید: `src/mosaic08_scale05_balanced_fold4_protocol.py`، `src/train_mosaic08_scale05_balanced_fold4.py`، `tests/test_mosaic08_scale05_balanced_fold4.py` و `configs/mosaic08_scale05_balanced_fold4.yaml`. فایل محلی مشترک `src/balanced_fitness_checkpoint.py` فقط برای metadata filename قابل‌تنظیم و subclass guard batch توسعه یافت. هیچ Ultralytics site-package، Run A، FULL169 یا fixed test تغییر/خوانده نشد.

`py_compile` پاس شد و تست نهایی CPU/static برابر `13 passed in 2.45s` بود. تست‌ها identity و isolation منبع، copy/load initialization، `resume=False` در شروع تازه، augmentation دقیق، 40 epoch، fitness و strict improvement، inherited بودن selection/early stopping عادی، hard-fail batch reduction و YAML بدون test را پوشش دادند. training شروع نشد.

## بررسی نتایج آموزش‌های تازه — 2026-09-16

در پاسخ به درخواست مشاهده نتیجه همه آموزش‌ها، artifactها و logهای موجود بررسی شدند. هیچ fixed-test result یا inference روی fixed test خوانده/اجرا نشد. برای آزمایش تک‌Fold MOSAIC08، inference جدید فقط روی 35 Study نگه‌داشته‌شده Fold 4 با `src/cache_full_study_oof.py` انجام شد: پنج مسیر `--weights` شامل Run A Foldهای 0 تا 3 و `best_balanced.pt` جدید برای Fold 4، همراه `--fold 4 --score-protocol deployment_score --imgsz 768 --batch 16 --device 0`. خروجی در `outputs/full_study_oof_mosaic08_scale05_balanced_fold4_deployment_score_cache/` ثبت شد. سایر Foldها inference نشدند.

### وضعیت واقعی اجراها

- `MOSAIC08_SCALE05_BALANCED_FOLD4`: آموزش 40 epoch با completion marker معتبر تمام شده؛ best عادی و balanced هر دو epoch 27. mAP50-95 جدید `0.06510` در برابر تاریخی Run A Fold 4 `0.07658` است. P جدید `0.19598` در برابر `0.34573`، R باکس جدید `0.25` در برابر `0.28333` است. source Run A Fold 4 hash همچنان `30e953e6f60b85b8e940559eb24c69b8817d46d27fcc3479b1fab0807c75ac16` است.
- `best.pt` و `best_balanced.pt` جدید hash فایل متفاوت ولی تمام 902 tensor مدل bitwise برابر دارند؛ selector متوازن checkpoint متفاوتی انتخاب نکرده است.
- `RUN_A_FULL169`: log نشان می‌دهد 59 epoch محاسباتی تمام شده و `results.csv` دارای 59 سطر با epochهای 1..59 و `weights/last.pt` موجود است. پس از آموزش، `_finish` به‌خاطر فرض اشتباه `last_epoch == FIXED_EPOCHS-1` خطا داده؛ مقدار واقعی CSV برابر 59 است. بنابراین `training_completed.json` و `weights/final_deployment.pt` ساخته نشده‌اند. mAPهای صفر در این run ناشی از `val` نحوی تک‌تصویری و بدون label هستند و معیار کیفیت مدل نیستند. این run هیچ held-out validation ندارد؛ برای آن امتیاز OOF/Macro-F1 مستقل ادعا نمی‌شود. در این نوبت هیچ checkpoint یا runner آن تغییر نکرد.
- `FITNESS_BALANCED_768` پنج‌Fold: هیچ‌کدام از پنج output fold وجود ندارد؛ این آزمایش شروع نشده است.

### ارزیابی paired full-study Fold 4

برای Run A Fold 4 و MOSAIC08 جدید، exact 35 Study/31 patient و توالی 890 SOP (شامل سه slice metadata-unknown) برابر بررسی شد؛ `imgsz=768`, `conf=0.01`, `NMS IoU=0.5` نیز برابر بود. برای هر دو cache، aggregator و آستانه **فریز‌شده از انتخاب Run A روی چهار Fold دیگر** اعمال شد: `max`, raw operating point=`0.443603515625`; هیچ threshold جدیدی روی Fold 4 انتخاب نشد. ICH/MLS از truth هستند، پس اعداد زیر `oracle_other_heads_macro_f1` توسعه‌ای‌اند، نه Macro-F1 نهایی submission.

| Fold 4 full-study | Run A | MOSAIC08 |
|---|---:|---:|
| oracle-other-heads Macro-F1 | 0.965899 | 0.974013 |
| Accuracy | 0.971429 | 0.971429 |
| fracture TP/FP/FN/TN | 3/1/2/29 | 1/0/4/30 |
| Sensitivity | 0.600 | 0.200 |
| Specificity | 0.966667 | 1.000 |
| Precision | 0.750 | 1.000 |
| Study PR-AUC، `max` | 0.708824 | 0.800000 |
| ROC-AUC، `max` | 0.890000 | 0.860000 |

Confusion matrix ترتیب `[0,1,2]`: Run A `[[8,1,0],[0,11,0],[0,0,15]]`؛ MOSAIC08 `[[9,0,0],[0,11,0],[0,1,14]]`. بهبود Macro-F1 صرفاً جابه‌جایی نوع یک خطا در 35 Study است، همراه با افت حساسیت شکستگی. Bootstrap paired stratified patient-clustered با 1000 تکرار: CI 95% delta Macro-F1 جدید منهای A `[-0.064147,0.102323]` و delta PR-AUC `[-0.065934,0.300000]`؛ هر دو صفر را قطع می‌کنند. با تنها پنج بیمار/Study مثبت، بهبود قطعی نیست و این single-Fold post-hoc experiment جایگزین انتخاب authoritative پنج‌Fold Run A نمی‌شود.

### مقایسه خانواده‌های دارای OOF کامل

نتایج موجود و از قبل کامل‌شده: Run A 768 `0.969627`، Run B 1024 `0.954958` و AUG_MILD 768 `0.938469` بر اساس `oracle_other_heads_macro_f1` patient-grouped full-study OOF. پس هنوز Run A بهترین خانواده اثبات‌شده است. B25 Fold 0 exploratory نیز حساسیت را افزایش نداد (TP=2 در هر دو) و FP را از 0 به 4 رساند. هیچ آموزش یا inference تازه برای این خانواده‌ها در این نوبت اجرا نشد.

محدودیت/قدم بعدی: قبل از استفاده از FULL169 برای deployment، خطای finalization باید جداگانه و بدون آموزش مجدد رفع و provenance checkpoint بررسی شود؛ این اقدام در درخواست صرفاً گزارش نتایج انجام نشد. MOSAIC08 فقط یک Fold دارد و نباید به‌عنوان بهبود رسمی پنج‌Fold پذیرفته شود. تست‌های نرم‌افزاری یا پایان process با بهبود مدل اشتباه گرفته نشدند.

## ترمیم نهایی‌سازی RUN_A_FULL169 بدون آموزش مجدد — 2026-09-16

**ایده و دلیل:** آموزش fixed-epoch روی تمام 169 Study توسعه‌ای به پایان رسیده بود، اما نشانگر تکمیل و checkpoint استقرار به‌دلیل خطای شماره‌گذاری epoch ساخته نشده بودند. هدف فقط اعتبارسنجی `last.pt` و نهایی‌کردن همان وزن بود؛ هیچ انتخاب checkpoint بر اساس داده آموزش/validation انجام نشد و `best.pt` مبنا قرار نگرفت.

**علت دقیق خطا:** `results.csv` دارای 59 ردیف با epochهای پیاپی `1..59` است، اما `_finish` مقدار ردیف آخر را با `FIXED_EPOCHS-1=58` مقایسه می‌کرد. در `last.pt`، `train_args.epochs=59` است؛ Ultralytics هنگام strip کردن optimizer، فیلد checkpoint `epoch` را به sentinel برابر `-1` تبدیل کرده است. تاریخچه داخلی `train_results` خود checkpoint نیز 59 ردیف `1..59` دارد و با کل `results.csv` دقیقاً برابر است. بنابراین epoch محاسباتی آخر zero-based برابر 58 است، ولی **CSV one-based برابر 59** و metadata checkpoint بعد از strip برابر `-1` است. لاگ پایان 59 epoch و سپس همان RuntimeError نهایی‌سازی را ثبت کرده بود.

**فایل‌های تغییرکرده:** فقط منطق finalization/CLI در `src/train_run_a_full169.py`، دو تست مصنوعی در `tests/test_run_a_full169_protocol.py` و این ثبت در `agent.md`. فایل‌های ایجادشده در output فقط `weights/final_deployment.pt` و `training_completed.json` هستند. معماری، config، داده، optimizer، augmentation، شمار epoch، `last.pt`، `best.pt` و `run_protocol.json` تغییر نکردند.

**روش ترمیم:** finalizer اکنون شماره‌های CSV را به‌صورت دقیق `1..59` بررسی می‌کند، protocol ذخیره‌شده و آرگومان‌های checkpoint را می‌سنجد، وجود مدل و optimizer-stripped بودن checkpoint را کنترل می‌کند و تمام تاریخچه داخلی checkpoint را با CSV مقایسه می‌کند. گزینه `--finalize-only` فقط همین مسیر CPU را فراخوانی می‌کند و به train/resume/inference وارد نمی‌شود. اگر artifact نهایی از قبل موجود باشد، از overwrite جلوگیری می‌شود. بعد از اعتبارسنجی، `final_deployment.pt` با `shutil.copy2` از `last.pt` ساخته و hash دو فایل مقایسه شد؛ completion marker وضعیت و provenance را ثبت می‌کند.

**دستور واقعی نهایی‌سازی:**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python src/train_run_a_full169.py --finalize-only
```

**تست‌های CPU/static:** `py_compile` روی runner و فایل تست پاس؛ دو تست synthetic finalization پاس (قبول تاریخچه one-based و checkpoint stripped؛ رد تاریخچه zero-based بدون ساخت artifact). اجرای `--status`، `completed_fixed_epoch_refit` را با `fixed_epochs=59` و `finalized_without_retraining=true` نشان داد. `sha256sum` برای هر دو فایل برابر است و `cmp -s` با exit code صفر، برابری بایت‌به‌بایت را تأیید کرد.

```text
last.pt SHA256             109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640
final_deployment.pt SHA256 109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640
```

**نتیجه و محدودیت:** checkpoint استقرار اکنون دقیقاً همان final-epoch `last.pt` است. این بررسی فقط صحت تکمیل نرم‌افزاری و provenance وزن را ثابت می‌کند؛ هیچ ادعایی درباره بهبود Macro-F1 یا کیفیت مدل از آن حاصل نمی‌شود. آموزش یا inference تازه اجرا نشد و fixed test بررسی نشد. قدم بعدی، فقط در صورت درخواست جداگانه، یکپارچه‌سازی وزن نهایی با submission و سنجش مستقل مجاز است.

## آماده‌سازی RUN_A_FRACTURE_CLASSIFIER_OOF — 2026-09-16

**فرضیه و دلیل:** خانواده منتخب Run A در OOF دارای TP=7، FP=3، FN=17، TN=142 و sensitivity=0.291667 بود. هدف آزمایش جداگانه این است که یک classifier برش، با featureهای backbone آموزش‌دیده آشکارساز، برخی FNهای Study را بازیابی کند بدون افزایش بزرگ FP. این مرحله فقط زیرساخت آزمایش را آماده می‌کند؛ بهبود مدل هنوز اندازه‌گیری نشده است. official-style `oracle_other_heads_macro_f1=0.969627` معیار baseline است، نه Macro-F1 نهایی submission، زیرا ICH/MLS در این تحلیل ground truth هستند.

**فایل‌های ایجادشده:** `src/fracture_classifier/{__init__,data,model,training,inference,compare}.py`، `tests/test_fracture_classifier.py`، `outputs/run_a_fracture_classifier_oof/slice_label_manifest.csv` و `outputs/run_a_fracture_classifier_oof/slice_label_audit.json`. این ثبت به `agent.md` اضافه شد. هیچ وزن Run A یا FULL169، کد detector، dataset آماده‌شده، `submit/model.py` یا site-packages تغییر نکرد.

**برچسب و cohort:** پنج فهرست validation تاریخی از طریق manifest به Study/patient نگاشت و با cache کاملِ OOF Run A با cohort دقیق 169 Study/155 patient تطبیق داده شد. فقط PNGهای حاصل از JSON annotation بازبینی‌شده برچسب 0/1 می‌گیرند؛ شمار باکس JSON، manifest و YOLO label با هم برابر بررسی می‌شود. برش بدون JSON در manifest جدید با label تهی و `ignored_no_annotation_json` ثبت می‌شود و وارد BCE نمی‌شود. نه همه برش‌های یک Study مثبت، مثبت فرض شدند و نه unknownها منفی فرض شدند. شمارش‌ها:

| Fold | مثبتِ بازبینی‌شده | منفیِ بازبینی‌شده | نادیده‌گرفته‌شده |
|---|---:|---:|---:|
| 0 | 46 | 854 | 14 |
| 1 | 46 | 831 | 7 |
| 2 | 45 | 824 | 0 |
| 3 | 43 | 842 | 0 |
| 4 | 43 | 844 | 3 |

مجموع 4418 برش review‌شده و 24 برش بدون JSON در همین cohort/caches؛ نشت بیمار بین Foldها صفر. 175 برش metadata-unknown قبلاً ممیزی‌شده با «24 برش بدون JSON» یک مفهوم نیستند: metadata-unknown می‌تواند همچنان JSON annotation داشته باشد. `pos_weight` در هر آموزش فقط از چهار Fold train محاسبه می‌شود.

**Provenance وزن‌های منبع، SHA256:** Foldهای 0 تا 4 به‌ترتیب `cb135cc18665e2348080990896f000aace8483d3153b79c293cc1c57cb23a395`، `bf144ac1c91799e0463b6ae0ec1b8bdd3f69d37377ff0e1e965f66044ada0757`، `6c3635082f4ed0e15d549e02e3a7faeaec0293299a06b01d342d8f2e3853f206`، `9a358eb448abe80d9364a20e47d57719fe43fc1eb44ffe2d1bb32f24bab0c5aa` و `30e953e6f60b85b8e940559eb24c69b8817d46d27fcc3479b1fab0807c75ac16` هستند. وزن محافظت‌شده FULL169 SHA256=`109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640` است. runner پیش و پس از آموزش، hash وزن‌های منبع را می‌سنجد و فقط پوشه مستقل `outputs/run_a_fracture_classifier_fold{k}/` می‌نویسد.

**معماری/آموزش فریز‌شده:** feature لایه 10 آشکارساز Run A، یعنی خروجی P5/32 بلوک `C2PSA` با 512 کانال؛ مسیر backbone لایه‌های 0..10 در checkpoint واقعی sequential است. head مستقل: Global Average Pooling → LayerNorm(512) → Linear(512,128) → SiLU → Dropout(0.10) → Linear(128,1). خروجی یک logit است؛ sigmoid فقط برای احتمال inference/validation به‌کار می‌رود. Stage 1 تمام backbone را منجمد و eval می‌کند؛ فقط head trainable است؛ AdamW، LR=1e-4، weight decay=5e-4، 10 epoch، weighted BCE با `neg_train/pos_train` و انتخاب `best_classifier.pt` صرفاً بر اساس slice validation PR-AUC. ورودی همان 2.5D ±5mm، WL=800/WW=1600، PNG 512×512 آماده Run A و resize به 768 است؛ تمام 4442 DICOM slice در cache 169 Study نیز 512×512 بودند. Mosaic، HU jitter، HNM، B25، 1024 و metadata negatives وارد نشده‌اند. Stage 2 اختیاری و پیش‌فرض خاموش است: فقط لایه 10 backbone باز می‌شود، head LR=1e-4 و لایه 10 LR=1e-5، حداکثر 5 epoch، خروجی‌های `best_stage2_classifier.pt` و `last_stage2_classifier.pt` جدا؛ OOF پیش‌فرض همچنان Stage 1 را می‌سنجد.

`pos_weight` فقط از train هر Fold، قبل از آموزش: Fold0=`3341/177=18.875706215`، Fold1=`3364/177=19.005649718`، Fold2=`3371/178=18.938202247`، Fold3=`3353/180=18.627777778`، Fold4=`3351/180=18.616666667`. این‌ها از manifest review‌شده‌اند و validation در محاسبه‌شان استفاده نشده است.

**نکته فنی کشف‌شده در تست:** checkpointهای stripped آشکارساز FP16 هستند. تست forward مصنوعی CPU روی backbone واقعی ابتدا به‌خاطر ورودی FP32/وزن FP16 خطا داد. فقط کپیِ درون حافظه classifier به FP32 تبدیل شد؛ وزن فایل منبع تغییر نکرد. همان تست سپس شکل خروجی `(2,)`، feature `C2PSA`، تعداد پارامتر backbone trainable برابر صفر و تعداد پارامتر head trainable برابر 66817 را تأیید کرد.

**Inference/مقایسه:** ماژول inference به‌صورت دستی فقط Studyهای صریحِ held-out Fold را می‌خواند و همه SOPهای cache کامل Run A را با DICOM اصلی/همان HU preprocessing به احتمال classifier تبدیل می‌کند؛ per-Study cache قابل ازسرگیری است. GPU کمتر از 8GiB آزاد، inference را قبل از model load رد می‌کند. مقایسه CPU از scoreهای authoritative Run A موجود استفاده می‌کند و detector را دوباره infer نمی‌کند. چهار aggregator از پیش مشخص classifier عبارت‌اند از max، top3_mean، top5_mean و top10_percent_mean؛ انتخاب هر Fold فقط از چهار Fold دیگر انجام می‌شود. برای fusion، alpha از grid ثابت `[0.25,0.50,0.75]` روی چهار Fold دیگر انتخاب می‌شود؛ threshold رسمی خروجی همچنان 0.5 است. خروجی شامل detector-only، classifier-only و weighted fusion، metricهای Study/fracture/triage، FP/FN نجات‌یافته، و paired patient-clustered bootstrap CI است. learned Study aggregator عمداً اضافه نشد زیرا هنوز evidence برای پیچیدگی آن موجود نیست.

**بررسی‌های انجام‌شده:** `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m py_compile src/fracture_classifier/*.py tests/test_fracture_classifier.py` پاس؛ `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_fracture_classifier.py` برابر **9 passed in 2.77s**؛ ممیزی واقعیِ فقط-development برچسب‌ها به‌صورت idempotent پاس شد. اعتبارسنجی عضویت Fold فقط از فهرست‌های development و ردیف‌های متناظر manifest استفاده می‌کند؛ هیچ شناسه/label از fixed test برای ساخت cohort استخراج نمی‌شود و `split=test` روی ردیف انتخاب‌شده hard-fail است. فرمان سریع CPU `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m fracture_classifier.compare --baseline-only` دقیقاً TP=7/FP=3/FN=17/TN=142، PR-AUC=0.535548638، ROC-AUC=0.729885057 و oracle Macro-F1=0.969627378 را بازتولید کرد. این بازتولید از artifact موجود است و classifier result یا GPU inference تازه نیست. CLIهای train/inference/compare با `--help` پاس شدند.

**محدودیت و قدم بعدی:** هنوز هیچ Fold classifier آموزش ندیده، هیچ OOF prediction جدید ساخته نشده و هیچ بهبود علمی ادعا نمی‌شود. اجرای بعدی فقط به‌صورت دستی توسط کاربر از Fold 0 آغاز می‌شود؛ پس از تکمیل همه پنج Fold، full-study classifier inference و مقایسه CPU باید اجرا شوند. OOF پیش‌فرض Stage 1 است؛ Stage 2 اختیاری نیز با `--stage stage2` در inference/comparator به cache/output مجزا می‌رود و هرگز خاموش جایگزین Stage 1 نمی‌شود. resume آموزش به‌خاطر نیاز به بازسازی دقیق optimizer/RNG هنوز پیاده نشده و نباید با راه‌اندازی تکراری یک خروجی نیمه‌کاره شبیه‌سازی شود. fixed test نه برای تصمیم/تنظیم و نه برای inference استفاده یا بررسی شد. هیچ GPU training یا GPU inference در این نوبت شروع نشد.

## آماده‌سازی refit نهایی FULL169 و handoff fusion — 2026-09-16

**فرضیه و علت کار:** آزمایش تکمیل‌شده پنج‌فولدی classifier در سطح fracture از TP=7/FP=3/FN=17/TN=142 به TP=8/FP=4/FN=16/TN=141، PR-AUC از 0.5355486381 به 0.5592862949 و sensitivity از 0.2916667 به 0.3333333 رسید، ولی oracle-other-heads triage Macro-F1 در هر دو 0.9696273781 ماند. هدف این نوبت فقط آماده‌کردن یک refit کامل، با طول ازپیش‌مشتق‌شده از OOF، و pipeline بستهٔ قابل‌تحویل است؛ **بهبود معیار اصلی مسابقه اثبات نشده** و oracle metric با Macro-F1 submission نهایی یکی نیست.

**ممیزی داده و epoch:** `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m fracture_classifier.train_full169 --audit` manifest ثابت classifier را دوباره از فهرست‌های صریح Run A و cache توسعه ساخت و برابری byte-level CSV را بررسی کرد. نتیجه: 169 Study، 155 بیمار، 223 برش مثبت review‌شده، 4195 برش منفی review‌شده، 24 برش unknown/بدون JSON که وارد BCE نمی‌شوند؛ `pos_weight=4195/223=18.81165919282511`. دادهٔ منتخب فقط splitهای توسعه است و ورود `split=test` hard-fail می‌شود؛ fixed test خوانده نشد. بهترین epochهای واقعی OOF Stage 1 برابر `[3,7,9,9,10]` و median برابر 9 است. seed=42 پیش از ساخت مدل، head-only AdamW با LR=1e-4، weight decay=5e-4، batch=8، weighted BCE و دقیقاً 9 epoch بدون validation/checkpoint selection برنامه‌ریزی شد. خروجی checkpoint آخر هر epoch در پوشهٔ تازه `outputs/run_a_fracture_classifier_full169/` است؛ finalizer فقط epoch نهایی را پس از بررسی metadata، manifest و SHA256 تأیید و `final_classifier.pt` را byte-identical از `last_classifier.pt` کپی می‌کند. تا این ثبت، وضعیت `not_started` است.

**قانون inference و محدودیت مهم:** هر پنج انتخاب cross-fitted classifier aggregator=`top5_mean` و alpha detector=`0.75` داشتند؛ قانون fusion برابر `clip(0.75*p_detector+0.25*p_classifier,0,1)` و threshold رسمی fracture همچنان 0.5 است. قانون deployment آشکارساز که قبلاً برای Run A روی همه 169 OOF Study ثبت شده، `top10_percent_mean` از score برش‌ها و نگاشت یکنوای raw operating point=`0.39717610677083337` به 0.5 است؛ conf=0.01، NMS=0.5 و imgsz=768. این selection سراسری **apparent/optimistic** است. OOF fusion گزارش‌شده از probabilityهای detector با transformationهای متفاوت cross-fitted هر Fold استفاده کرده بود؛ بنابراین عدد 0.969627/0.559286 اثبات مستقیم عملکرد همین mapping سراسری به‌همراه classifier FULL169 نیست. با وجود این، mapping سراسری از قبل در Run A deployment ثبت شده بود و در این نوبت قانون جدید از fixed test یا tuning تازه ابداع نشد.

**ناهمسانی کانال کشف‌شده:** `ReviewedSlices` در آموزش OOF و refit، PNG را با OpenCV می‌خواند و BGR→RGB می‌کند، در حالی که inference تکمیل‌شده OOF خروجی مستقیم `make_hu_input` را بدون برگرداندن کانال به `image_tensor` می‌دهد. این تفاوت در یک برش development با مقایسهٔ آرایه‌ها تأیید شد. handoff عمداً همان مسیر inference واقعی OOF را حفظ می‌کند تا protocol خاموش تغییر نکند؛ این محدودیت می‌تواند کیفیت را تحت تأثیر بگذارد و نباید به‌عنوان روش بهینه تلقی شود. رفع آن نیازمند آزمایش OOF مستقل است، نه دستکاری بی‌صدای بستهٔ نهایی.

**فایل‌ها و محافظت:** `src/fracture_classifier/train_full169.py`، `build_handoff.py`، `handoff_runtime.py` و `handoff_benchmark.py` اضافه/تکمیل و `tests/test_fracture_classifier_full169_handoff.py` اضافه شد. سازندهٔ بسته فقط پس از finalization، `handoff/fracture_run_a_classifier_fusion_v1/` جدید را می‌سازد، دو وزن واقعی را کپی و SHA256 را کنترل می‌کند؛ اگر مقصد موجود باشد یا وزن classifier نهایی وجود نداشته باشد، fail-closed است. predictor فقط یک `fracture_prob` بازمی‌گرداند و triage نهایی/ICH/MLS را اجرا نمی‌کند. تمام DICOMهای سری هدف بدون نیاز به JSON، با ترتیب فیزیکی، HU، MONOCHROME1، WL800/WW1600 و 2.5D ±5mm خوانده می‌شوند؛ در حضور چند سری، UID هدف باید صریح داده شود. دو مدل جداگانه در batchهای قابل‌تنظیم اجرا می‌شوند؛ GPU memory/runtime هنوز benchmark نشده‌اند. benchmark دستی count مطالعه/برش، زمان و peak CUDA memory را گزارش می‌کند.

**آزمون و نتیجهٔ واقعی:** `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m py_compile` برای چهار ماژول تازه پاس شد. `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_fracture_classifier_full169_handoff.py` در اجرای اولیه 6/6 پاس شد؛ پس از افزودن دو assertion محافظتی، suite دوباره اجرا و نتیجه در ادامه ثبت می‌شود. تست مصنوعی قرارداد `predict_study`، top5/fusion، freeze/head architecture، قانون OOF، cohort واقعی و عدم overwrite را سنجید. این‌ها **تست نرم‌افزاری‌اند، نه بهبود مدل**. SHA256 detector FULL169 در audit برابر `109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640` بود. هیچ checkpoint اصلی، `submit/model.py`، dataset یا site-packages تغییر نکرد؛ هیچ آموزش/GPU inference اجرا نشد؛ بستهٔ handoff هنوز ساخته نشده چون classifier نهایی هنوز آموزش ندیده است.

**قدم بعدی:** کاربر به‌صورت دستی آموزش 9 epoch را در tmux اجرا کند؛ پس از تکمیل واقعی، CPU-only finalization و سپس CPU-only packaging را اجرا کند؛ smoke inference و benchmark تنها در زمان مناسب و جدا از این نوبت اجرا شوند. قبل از ادعای بهبود مسابقه، ارزیابی مستقل کامل با ICH/MLS پیش‌بینی‌شده لازم است؛ fixed test برای انتخاب یا تنظیم استفاده نشود.

**نتیجهٔ نهایی تست همین نوبت:** پس از تکمیل assertionها، `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_fracture_classifier_full169_handoff.py` برابر **7 passed in 6.94s** شد. `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m py_compile` برای چهار ماژول و فایل تست پاس شد. `--status` وضعیت `not_started` را برای خروجی FULL169 classifier گزارش کرد؛ بستهٔ handoff نیز اجرا/ساخته نشد. این نتایج فقط سلامت کد و provenance ورودی را نشان می‌دهند، نه دقت checkpointی که هنوز تولید نشده است.

**تقویت تست freeze:** تست معماری پس از این ثبت به‌جای backbone مصنوعی، backbone واقعی وزن FULL169 detector را روی CPU بارگذاری کرد و وجود پارامترهای backbone، freeze کامل آن و trainable بودن head را سنجید. اجرای دوبارهٔ همان suite برابر **7 passed in 7.13s** شد. هیچ GPU inference/training انجام نشد.

**تست finalization مصنوعی:** با checkpoint ساختگی صرفاً در tmp، finalizer تأیید epoch نهم، برابری byte-level کپی `last_classifier.pt`/`final_classifier.pt` و رد اجرای دوم را گذراند؛ این checkpoint ساختگی کیفیت مدل ندارد و وارد خروجی واقعی نشد. کل suite اکنون **8 passed in 7.32s** است. وزن واقعی classifier هنوز ساخته نشده و هیچ completion واقعی ادعا نمی‌شود.

## به‌روزرسانی submit برای تحویل fracture — 2026-09-16

**هدف و تصمیم بر مبنای artifact واقعی:** پوشهٔ موجود پروژه `submit/` بود، نه یک submission کامل چند-head. `submit/model.py` از ابتدا هفت کلید رسمی را برمی‌گرداند ولی پنج مقدار ICH و `MLS_mm` در آن صفر placeholder بودند؛ هیچ مدل واقعی ICH/MLS در این پوشه وجود نداشت. `outputs/run_a_fracture_classifier_full169/final_classifier.pt` و `training_completed.json` در زمان کار وجود نداشتند؛ بنابراین fusion هرگز فعال یا شبیه‌سازی نشد. بهترین انتخاب deployable موجود، detector نهایی Run A FULL169 با 59 epoch ثابت بود. فرضیهٔ این تغییر «بهتر شدن امتیاز» نیست؛ هدف حذف انتخاب غیرقابل‌دفاع Fold 2 از بسته و تحویل وزن نهایی خانوادهٔ منتخب همراه provenance دقیق است.

**پشتیبان و تغییر وزن:** پیش از تغییر، کپی کامل `submit_backup_before_full169_20260916/` ساخته و برابری فایل‌های `model.py` و `README.md` و هش وزن Fold 2 بررسی شد. SHA256 وزن قدیمی Fold 2=`6c3635082f4ed0e15d549e02e3a7faeaec0293299a06b01d342d8f2e3853f206` در backup محفوظ است. `submit/models/best.pt` با کپی byte-identical از `outputs/yolo26s_p2_hu800_ww1600_run_a_full169/weights/final_deployment.pt` جایگزین شد؛ هش هر دو `109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640` است. منبع detector تغییر نکرد. این بازنویسی وزن فعال عمدی و پس از backup بود، نه خاموش.

**کد inference:** در `submit/model.py` فقط docstring وزن و مسیر خواندن Study برای fracture به‌روز شد. همان HU، MONOCHROME1، WL800/WW1600، 2.5D فیزیکی ±5mm، imgsz768، conf0.01، NMS0.5، تجمیع `top10_percent_mean` و نگاشت یکنوای raw `0.39717610677083337`→0.5 باقی ماند. اکنون تمام DICOMهای سری هدف بدون نیاز به JSON خوانده می‌شوند؛ اگر چند SeriesInstanceUID وجود داشته باشد و UID هدف صریح داده نشود، مسیر fail-closed می‌شود تا سری‌های متفاوت خاموش مخلوط نشوند. اگر Study فقط یک سری داشته باشد API قدیمی `Model().predict(study_dir)` تغییری نمی‌خواهد؛ برای چند سری `Model(target_series_uid=...).predict(...)` لازم است. هفت کلید و placeholderهای ICH/MLS عوض نشدند.

**مستندات و مرجع آموزش:** `submit/README.md` و `SELECTION_REPORT.json` وضعیت active FULL169/no fusion را جایگزین ادعای قدیمی Fold 2 کردند. `MODEL_PROVENANCE.md` و `SHA256SUMS.txt` افزوده شدند. پوشهٔ `submit/training_reference/` فقط کپی متن معماری YOLO26s-P2، `run_protocol.json`، `args.yaml`، runner/protocol/init/preparation source و `fracture_postprocessing.json` دارد؛ datasets، OOF caches، logs، checkpointهای optimizer و fixed-test predictionها کپی نشدند. کد classifier به‌عنوان dependency فعال گنجانده نشد، چون final classifier موجود نبود.

**شواهد OOF و محدودیت:** artifact پنج‌فولدی موجود detector-only را TP7/FP3/FN17/TN142، sensitivity0.2916667، Study PR-AUC0.5355486، fracture F1=0.4117647 گزارش کرده؛ fusion آزمایشی را TP8/FP4/FN16/TN141، sensitivity0.3333333، PR-AUC0.5592863، ROC-AUC0.7672414، fracture F1=0.4444444. oracle-other-heads triage Macro-F1 برای هر دو `0.9696273781` است. این metric از ICH/MLS واقعی بهره می‌برد و نه Macro-F1 submission نهایی است و نه معیار اندازه‌گیری‌شدهٔ refit FULL169. قانون fusion آزمایشی top5_mean و alpha 0.75 فقط در provenance آمده، نه در inference فعال. هیچ ادعای بهبود Macro-F1 مسابقه نشده است.

**تست‌های واقعی این نوبت:** `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_submit_full169_static.py` برابر **3 passed in 0.49s**. تست‌ها hash/وجود وزن، import بدون YOLO/GPU، تنظیمات preprocessing، خروجی هفت‌کلیدی و probability در [0,1]، نقطهٔ calibration و خواندن تمام برش‌های یک سری/رد اختلاط دو سری را با دادهٔ مصنوعی بررسی کردند. `py_compile` برای `submit/model.py` و چهار فایل مرجع Python و تست پاس شد؛ pycهای تولیدیِ قابل‌بازسازی پس از تست از بستهٔ `submit/` حذف شدند (اصل آن‌ها در backup باقی است). `sha256sum -c submit/SHA256SUMS.txt` تمام 14 فایل فهرست‌شده را OK گزارش کرد. فهرست نهایی بسته بررسی شد: هیچ فایل data، cache، log یا prediction ثابت-test وجود ندارد. **هیچ آموزش یا GPU inference و هیچ smoke واقعی DICOM در این نوبت اجرا نشد.**

**قدم بعدی:** هم‌تیمی ابتدا باید مدل‌های واقعی ICH/MLS را به placeholderهای هفت‌کلیدی وصل کند و سپس smoke واقعی را روی یک Study مجاز اجرا کند. تا پیش از آن، این بسته تنها component شکستگی است، نه submission کامل triage. اگر classifier FULL169 بعداً واقعاً نهایی شد، fusion باید در تغییر جداگانه با وزن/hash و validation مناسب اضافه شود؛ absence امروز مجوز استفاده از checkpoint OOF به‌عنوان جایگزین نیست.

## آماده‌سازی ارزیابی apparent سیستم نهایی FULL169 — 2026-09-16

**هدف و مرز تفسیر:** درخواست جدید اندازه‌گیری detector نهایی، classifier نهایی و fusion روی همان 169 Study آموزشیِ FULL169 است. این نتیجه فقط `apparent_training_cohort_performance` خواهد بود: چون هر دو وزن روی همین cohort آموزش دیده‌اند، نه validation مستقل، نه OOF و نه fixed test است. OOF fusion موجود با TP8/FP4/FN16/TN141، sensitivity0.3333333، specificity0.9724138، precision0.6666667، F1=0.4444444، PR-AUC0.5592863، ROC-AUC0.7672414 تنها مرجع unbiased پنج‌فولدی و جداگانه در گزارش آینده خواهد بود. oracle-other-heads triage Macro-F1 آن 0.9696273781 است؛ این هم Macro-F1 نهایی submission نیست.

**واقعیت checkpoint و cohort:** اکنون `outputs/run_a_fracture_classifier_full169/final_classifier.pt` و `training_completed.json` واقعاً وجود دارند؛ SHA256 classifier=`ec0d34c7164453a8a80e476762080829bb4f8b219fd5fc8067380af891a50485` و detector=`109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640`. پیش‌ممیزی خواندنیِ CPU با فرمان `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m fracture_classifier.evaluate_full169 --preflight` پاس شد: 169 Study، 155 بیمار، 24 Study مثبت، 145 Study منفی، Fold counts `[33,33,32,36,35]` و هم‌پوشانی Study/patient با fixed test صفر. این ممیزی از Studyهای صریح validation تاریخی و manifest نگاشت می‌گیرد؛ هیچ fixed-test DICOM یا prediction خوانده/ایجاد نشد. پوشهٔ `evaluation/` هنوز وجود ندارد.

**پیاده‌سازی:** `src/fracture_classifier/evaluate_full169.py` افزوده شد. قبل از model load، هش دو checkpoint، marker تکمیل 9 epoch classifier، protocol، cohort، patient mapping و ثابت‌بودن تنظیمات `submit/model.py` را بررسی می‌کند. `--preflight` فقط CPU است. حالت دستی `--evaluate` با guard حداقل 8 GiB VRAM، header guard تاریخی target CT series را روی همان 169 Study به کار می‌برد؛ همهٔ برش‌های معتبر سری هدف حتی metadata-unknown بدون نیاز به JSON infer می‌شوند. DICOM از نظر فیزیکی مرتب، به HU تبدیل، MONOCHROME1 رسیدگی، WL800/WW1600 و زمینهٔ 2.5D ±5mm ساخته می‌شود. detector از `submit/model.py` با imgsz768/conf0.01/NMS0.5 فراخوانی می‌شود. classifier همان convention inference تاریخی OOF، یعنی `image_tensor` روی آرایهٔ مستقیم HU سه‌کاناله، را مصرف می‌کند؛ ناهمسانی کانال نسبت به PNG train که در بخش قبل ثبت شده همچنان تغییر نکرده است. detector probability با `top10_percent_mean` و نگاشت raw `0.39717610677083337` به 0.5 از خود submit گرفته می‌شود؛ classifier با `top5_mean` و fusion با `clip(0.75*detector+0.25*classifier,0,1)` محاسبه می‌شود. همهٔ تصمیم‌ها در threshold رسمی 0.5 هستند.

**خروجی/ایمنی اجرا:** پیش‌بینی‌ها قبل از اتصال truth، به‌صورت قابل‌ادامه در `evaluation/study_cache/<Study>.json` با SOP list، signature و احتمال‌ها ذخیره می‌شوند. cache با Study/patient/signature/SOP/slice count/fusion equation نامنطبق یا cache اضافه خارج cohort رد می‌شود. ground-truth `SkullFracture` فقط پس از تکمیل همه 169 predictionها به جدول وصل می‌شود؛ در `--preflight` صرفاً برای audit CPU شمار cohort خوانده شد، نه برای inference. خروجی نهایی برنامه‌ریزی‌شده: `study_predictions.csv`، `report.json` با سه سیستم و برچسب آشکار apparent، `confusion_matrix.csv` و PNG confusion matrix fusion با محورهای Actual/Predicted و اعداد خام. اگر report نهایی موجود باشد، اجرای دوباره آن را بازنویسی نمی‌کند. هیچ وزن مدل یا submit تغییر نکرد و هیچ آموزش/GPU inference شروع نشد.

**تست‌ها و خطای رفع‌شده:** `tests/test_fracture_classifier_full169_evaluator.py` افزوده شد؛ top5/fusion/calibration، برابری helper فعلی submit با HU/context تاریخی OOF روی داده مصنوعی، معیارها و جهت matrix، reject cache ناهمخوان، و ساخت چهار خروجی با cohort مصنوعی 169تایی را می‌سنجد. اجرای مستقل این فایل **5 passed in 2.91s** بود. اجرای مشترک اولیه با آزمون قدیمی handoff به‌دلیل وجود واقعی پوشهٔ handoff با 1 failure تمام شد؛ آن آزمون به‌اشتباه نبودن پوشهٔ handoff واقعی را پیش‌فرض گرفته بود. فقط fixture تست در `tests/test_fracture_classifier_full169_handoff.py` اصلاح شد تا destination موقت مستقل به کار گیرد؛ اجرای مجدد دو فایل **13 passed in 8.10s** شد. `py_compile` برای evaluator و دو فایل تست پاس شد. این‌ها تست نرم‌افزاری‌اند، نه metric مدل. SHA256 هر دو وزن پس از تست بدون تغییر ماند. پوشهٔ `outputs/run_a_fracture_classifier_full169/evaluation/` هنوز ایجاد نشده است.

**فرمان دستیِ مرحلهٔ بعد، نه اجراشده در این نوبت:** از ریشهٔ پروژه `PYTHONPATH=src .venv/bin/python -u -m fracture_classifier.evaluate_full169 --evaluate --device cuda:0 --batch 8`. بعد از اتمام واقعی باید گزارش را با برچسب apparent خواند و از مقایسهٔ مستقیم آن با OOF به‌عنوان improvement پرهیز کرد. fixed test همچنان خارج از محدوده است.

**تقویت provenance و نتیجهٔ نهایی تست:** evaluator علاوه بر hashِ `final_classifier.pt`، byte identity با `last_classifier.pt` را در preflight بررسی می‌کند و هنگام model load، epoch=9، fixed_epochs=9، stage=1، hash detector و hash manifest آموزشی داخل checkpoint را می‌سنجد. preflight پس از این تغییر دوباره پاس شد و همچنان همان 169/155/24/145 و صفر هم‌پوشانی را گزارش کرد. اجرای نهایی دو suite CPU برابر **13 passed in 7.90s** و `py_compile` پاس بود. `evaluation/` هنوز ساخته نشده است؛ هیچ training یا GPU inference انجام نشده و هیچ metric ظاهری FULL169 هنوز وجود ندارد.

## ممیزی ۱۷ FN و آزمون OOF کالیبراسیون/gated fusion — 2026-09-16

**هدف، فرضیه و مرز داده:** نتیجهٔ FULL169 روی همان ۱۶۹ Study آموزشی، هرچند TP=23/FP=0/FN=1/TN=145 و PR-AUC=1.0 دارد، ظاهری/in-sample است و در این آزمایش برای انتخاب هیچ پارامتر یا مدل استفاده نشد. هدف، بررسی ۱۷ FN واقعیِ Run A در OOF بیمارمحور و آزمودن این فرضیه بود که logistic meta-score یا gate می‌تواند با FP قابل‌قبول FNها را بازیابی کند. فقط artifactهای OOF موجود خوانده شدند؛ هیچ inference یا آموزش جدیدی انجام نشد. معیار triage در اینجا `oracle_other_heads_macro_f1` است: ICH و MLS واقعی فقط هنگام ارزیابی rule رسمی وارد می‌شوند، نه به‌عنوان feature مدل؛ این معیار **Macro-F1 واقعی submission نیست**.

**پیش‌شرط بازتولید:** قبل از پیاده‌سازی، فرمان `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m fracture_classifier.compare --baseline-only` دقیقاً Run A را بازتولید کرد: TP=7، FP=3، FN=17، TN=142، PR-AUC=`0.5355486380673358`، ROC-AUC=`0.7298850574712643`، oracle Macro-F1=`0.9696273781380164`. برنامهٔ جدید نیز پیش از ادامه همین اعداد را با tolerance عددی سخت چک می‌کند. cohort دقیقاً ۱۶۹ Study، ۱۵۵ بیمار، ۲۴ مثبت، ۱۴۵ منفی و به‌ترتیب `[33,33,32,36,35]` Study در Foldهای ۰ تا ۴ است. هر بیمار فقط در یک Fold قرار دارد؛ هم‌پوشانی Study/patient با test ثابت صفر است. هیچ برش fixed test یا نتیجهٔ آن برای تحلیل باز نشد.

**تعریف detector و احتیاط مهم leakage:** در submission فعلی یک transform یکتای deployable موجود است: `top10_percent_mean` سپس نگاشت یکنوای raw point=`0.39717610677083337` به مرز رسمی ۰٫۵. همان transform بدون انتخاب aggregator جدید در F0/F1/F2 به کار رفت. اما raw point در یک مرحلهٔ تاریخی با نگاه به **تمام OOF** انتخاب شده بود؛ بنابراین هرچند C، ضرایب logistic و tau در این آزمایش فقط از چهار Fold منبع انتخاب/fit شده‌اند، نتیجهٔ F0/F1/F2 *مشروط به نگاشت قبلاً apparent* است و برآورد strict untouched-protocol/unbiased نامیده نمی‌شود. ستون تاریخی `Run A detector` و `existing weighted fusion` قانون fold-specific قدیمی را بازتولید می‌کنند؛ `F0 fixed transform` comparator سازگار با تعریف فعلی deployment است، نه همان probability تاریخی.

**پیاده‌سازی و فایل‌ها:** `src/fracture_classifier/calibration.py` برای logit با clipping `1e-5`، LogisticRegression کوچک L2 با `class_weight=None` و grid `C={0.1,1,10}` اضافه شد. C بر اساس کمینهٔ log loss در inner held-out Foldهای *داخل چهار Fold منبع* انتخاب می‌شود؛ سپس مدل فقط روی همان چهار Fold fit می‌شود. `src/fracture_classifier/gated_fusion.py` قانون دقیق gate را دارد: اگر detector>=0.5، همان detector؛ اگر `tau_low<=detector<0.5`، meta؛ وگرنه detector؛ `tau_low` فقط از `{0,.05,.10,.15,.20,.25,.30,.35,.40}` و با معیار oracle Macro-F1 سپس FP/specificity/sensitivity/PR-AUC روی inner source predictionها انتخاب می‌شود. threshold خروجی همواره ۰٫۵ است. `src/fracture_classifier/compare_fusion_v2.py` provenance، audit برش‌ها، نمودارها، metricها، ۵ outer Fold و paired patient-clustered bootstrap طبقه‌بندی‌شده با ۵۰۰۰ تکرار/seed=`20260916` را اجرا می‌کند؛ در صورت وجود خروجی قبلی overwrite را رد می‌کند. `tests/test_fracture_classifier_fusion_v2.py` تست‌های CPU اضافه‌شده است. وزن‌ها، `submit/`، داده و پروتکل train تغییر نکردند.

**ممیزی FN و توزیع:** دسته‌ها: ۱۲ مورد `detector_low_classifier_high`، چهار مورد `near_threshold` و یک مورد `both_low`. جدول زیر scoreهای تاریخی OOF است، نه FULL169:

| Study | detector | classifier top5 | fusion تاریخی | دسته |
|---|---:|---:|---:|---|
| 272626 | 0 | .603144 | .150786 | low/high |
| 4066 | 0 | .794333 | .198583 | low/high |
| 1843 | 0 | .587476 | .146869 | low/high |
| 271623 | 0 | .627171 | .156793 | low/high |
| 7317 | 0 | .381174 | .095293 | both low |
| 3206 | .006862 | .656516 | .169276 | low/high |
| 3905 | .016515 | .758610 | .202039 | low/high |
| 270845 | .020451 | .807020 | .217093 | low/high |
| 342691 | .046099 | .761940 | .225060 | low/high |
| 3416 | .049483 | .677635 | .206521 | low/high |
| 9244 | .070515 | .651532 | .215769 | low/high |
| 4240 | .102346 | .820942 | .281995 | low/high |
| 7150 | .299670 | .754633 | .413411 | low/high |
| 2265 | .300584 | .595864 | .374404 | near |
| 7203 | .315951 | .888957 | .459203 | near |
| 342682 | .454166 | .676559 | .509765 | near |
| 6709 | .482806 | .360873 | .452323 | near |

توزیع `min/p10/p25/median/p75/p90/max` به‌ترتیب: detector مثبت `0/0/.014102/.201008/.605339/.825519/.929794` و منفی `0/0/0/.010402/.047090/.153176/.667398`؛ classifier مثبت `.360873/.569657/.621164/.678609/.770070/.803214/.888957` و منفی `.293664/.446495/.505625/.635372/.708580/.776986/.886535`؛ fusion تاریخی مثبت `.095293/.152588/.201175/.328200/.625382/.805797/.888974` و منفی `.077155/.120184/.149938/.177433/.210734/.273785/.690606`. بنابراین min مثبت/max منفی به‌ترتیب detector `0/.667398`، classifier `.360873/.886535` و fusion `.095293/.690606` است؛ هم‌پوشانی score شدید، به‌خصوص برای classifier، دیده می‌شود. این نمودارها فقط diagnostic هستند.

**اثر تصمیم triage:** از ۱۷ FN، تبدیل fracture از منفی به مثبت فقط در ۳ Study (`271623`، `272626`، `4240`) برچسب oracle triage را عوض می‌کند؛ در هر سه مورد `1 -> 2` است. ۱۴ مورد تغییری در triage ندارند. این تحلیل از ICH/MLS واقعی فقط برای سنجش impact استفاده می‌کند، نه به‌عنوان feature یا انتخاب مستقیم threshold روی held-out Fold.

**نتیجهٔ پنج‌فولدی pooled با threshold رسمی ۰٫۵:**

| سیستم | TP/FP/FN/TN | sens/spec/precision/F1 | PR-AUC / ROC-AUC | oracle Macro-F1 | triage CM ردیف truth 0/1/2 |
|---|---|---|---|---:|---|
| Run A تاریخی | 7/3/17/142 | .2917/.9793/.7000/.4118 | .535549/.729885 | .969627 | `[[22,1,0],[0,50,1],[0,3,92]]` |
| fusion تاریخی 0.75/0.25 | 8/4/16/141 | .3333/.9724/.6667/.4444 | .559286/.767241 | .969627 | همان Run A |
| F0 با transform ثابت | 8/2/16/143 | .3333/.9862/.8000/.4706 | .587270/.767241 | .974666 | `[[22,1,0],[0,51,0],[0,3,92]]` |
| F1 logistic | 0/0/24/145 | 0/1/تعریف‌نشده/0 | .489343/.719253 | .975415 | `[[23,0,0],[0,51,0],[0,5,90]]` |
| F2 gated | 8/0/16/145 | .3333/1/1/.5000 | .608142/.750287 | .985129 | `[[23,0,0],[0,51,0],[0,3,92]]` |

balanced accuracy برای پنج ردیف بالا به‌ترتیب `.635489/.652874/.659770/.500000/.666667`؛ Brier `.094376/.097455/.094432/.105801/.088683`؛ log loss `1.258661/.347669/.341825/.378501/.691410` است. classwise triage F1 برای Run A/تاریخی `[.977778,.952381,.978723]`، F0 `[.977778,.962264,.983957]`، F1 `[1,.953271,.972973]` و F2 `[1,.971429,.983957]` بود. Logistic مستقیم در مرز ۰٫۵ همهٔ fractureها را منفی کرد؛ بالا بودن oracle Macro-F1 آن نشانهٔ کارایی fracture نیست.

**پارامترهای held-out:** `(Fold,C,tau_low)` برای ۰ تا ۴ به‌ترتیب `(0,.1,0)`، `(1,.1,0)`، `(2,1,.15)`، `(3,1,0)`، `(4,1,0)` بود. در OOF نتیجه‌شده، بیشینهٔ `p_meta=.486998063` است و **هیچ meta score به ۰٫۵ نرسید**. بنابراین F2 در سطح تصمیم دقیقاً همان detector با transform ثابت است؛ هیچ rescue تصمیمی از gate رخ نداد. نسبت به Run A تاریخی، Study `6709` از FN به TP رفت، ولی همین TP در F0 ثابت هم موجود است (`detector_fixed=.500844`؛ detector تاریخی `.482806`) و محصول transform قبلاً انتخاب‌شده است، نه logistic/gate. F2 نسبت به F0 دو FP ناشی از fusion (Studies `1938` و `7396`) را حذف کرد؛ TP را افزایش نداد. نسبت به Run A تاریخی، F2 سه FP را حذف و هیچ FP تازه‌ای اضافه نکرد. PR-AUC بالاتر F2 عمدتاً ranking پیوستهٔ متفاوت است و با بهبود recall در ۰٫۵ یکی نیست.

**bootstrap paired در سطح بیمار، ۵۰۰۰ تکرار، seed=20260916:** بازه‌های ۹۵٪ برای delta نسبت به Run A تاریخی، به‌ترتیب `sensitivity / specificity / PR-AUC / fracture F1 / oracle Macro-F1`:

| سیستم | Δ sens CI | Δ spec CI | Δ AP CI | Δ F1 CI | Δ oracle Macro-F1 CI |
|---|---|---|---|---|---|
| fusion تاریخی | `[0,.13043]` | `[-.02158,0]` | `[-.00508,.05551]` | `[-.03158,.14510]` | `[0,0]` |
| F0 ثابت | `[0,.13043]` | `[0,.02113]` | `[.00347,.10594]` | `[0,.17037]` | `[0,.01631]` |
| F1 logistic | `[-.50000,-.08696]` | `[0,.05517]` | `[-.17594,.10118]` | `[-.62857,-.16000]` | `[-.01758,.03604]` |
| F2 gated | `[0,.13043]` | `[0,.05517]` | `[-.01000,.16687]` | `[0,.20509]` | `[0,.04299]` |

**Verdict از معیارهای پیش‌ثبت‌شده: `REJECT`.** F2 TP=8، FN=16 و sensitivity=.3333 دارد؛ شرط TP>=10، FN<=14 و sensitivity>=.417 برقرار نیست. بهبود apparent/conditional oracle Macro-F1 از `.969627` به `.985129` و حذف FP واقعی در همین ۱۶۹ Study، شرط ناکافی برای GO است؛ CI مربوط به delta Macro-F1 `[0,.04299]` نیز lower bound مثبت ندارد. gate هیچ FN تازه‌ای نسبت به fixed detector نگرفت. پس وزن یا inference قانون submission فعلی با F2 جایگزین نمی‌شود. آزمایش پیشنهادی بعدی طبق تصمیم پیشینی، classifier منجمد چندمقیاسی P3+P4+P5 است، نه epoch بیشتر برای classifier تک‌لایهٔ P5؛ این پیشنهاد فعلاً اجرا نشده است.

**خروجی‌های واقعی و روش اجرا:** `outputs/run_a_fracture_classifier_oof/fn_audit/` شامل `fn_audit.csv`، `fn_summary.json`، `score_distribution.csv`، `decision_impact_audit.csv`، `fn_visual_review_manifest.csv` و سه PNG histogram است. `outputs/run_a_fracture_classifier_oof/fusion_v2/` شامل `cross_fitted_study_predictions.csv` و `report.json` است. فرمان اجرا: `MPLCONFIGDIR=/tmp/matplotlib_fracture_v2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -u -m fracture_classifier.compare_fusion_v2 --bootstrap-repeats 5000 --seed 20260916`. خروجی‌ها کامل ساخته شدند؛ این صرفاً CPU post-processing روی cache قبلی بود.

**تست و ایمنی:** `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_fracture_classifier_fusion_v2.py` برابر **5 passed**؛ اجرای تکراری با thread limit نیز **5 passed**. `py_compile` برای سه فایل source و فایل تست پاس شد. تست نرم‌افزاری به معنی improvement مدل نیست. SHA256 محافظت‌شدهٔ detector FULL169=`109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640` و classifier FULL169=`ec0d34c7164453a8a80e476762080829bb4f8b219fd5fc8067380af891a50485` پس از تحلیل بدون تغییر بودند؛ وزن‌های OOF Run A نیز با hashهای ثبت‌شدهٔ قبلی برابر ماندند. هیچ فایل داخل `submit/` تغییر نکرد؛ هیچ fixed-test inference/inspection، GPU training/inference یا neural training جدید انجام نشد.

## آماده‌سازی ablation منجمد P3+P4+P5 — 2026-09-16

**فرضیه و دلیل:** آزمایش logistic/gated قبلی به‌دلیل TP=8/FN=16 رد شد؛ ۱۲ مورد از ۱۷ FN آشکارساز با score پایین آشکارساز ولی classifier بالا بودند، اما head تک‌مقیاسی P5 نجات واقعی در مرز رسمی نداد. فرضیهٔ جدید این است که افزودن ویژگی‌های ظریف‌تر P3 و P4 به **همان** P5 موجود، طبقه‌بندی presence شکستگی را بهتر کند. این یک ablation معماری است: داده، 2.5D، HU، window، آموزش Stage 1، optimizer، loss، انتخاب checkpoint و fusion اولیه عوض نمی‌شوند. هیچ claim بهبود مدل تا پیش از تکمیل پنج Fold و OOF وجود ندارد.

**گراف واقعی، نه فرض generic:** همهٔ پنج checkpoint واقعی Run A روی CPU بارگذاری و graph audit شدند. backbone لایه‌های ۰..۱۰ sequential (`f=-1`) است؛ P3 لایهٔ ۴=`C3k2`، stride=8، channels=256؛ P4 لایهٔ ۶=`C3k2`، stride=16، channels=256؛ P5 لایهٔ ۱۰=`C2PSA`، stride=32، channels=512. Detect layer 29 شاخه‌های `[19,22,25,28]` با strideهای `[4,8,16,32]` دارد. انتخاب از *backbone* عمدی است: P5 لایهٔ ۱۰ همان representation classifier قدیمی می‌ماند؛ انتخاب neck لایه‌های ۲۲/۲۵/۲۸ خود P5 را هم عوض می‌کرد و اثر «افزودن مقیاس» را مخدوش می‌کرد. synthetic forward واقعی Fold 0 با tensor `1×3×768×768` روی CPU این شکل‌ها را تأیید کرد: P3=`1×256×96×96`، P4=`1×256×48×48`، P5=`1×512×24×24`. hookها در هر forward تعداد فراخوانی، module type، channels، spatial size و stride را hard-fail می‌کنند اگر نامنطبق باشند.

**معماری دقیق:** روی هر یک از P3/P4/P5، `GAP -> LayerNorm(C) -> Linear(C,128) -> SiLU`؛ concat سه خروجی ۱۲۸تایی به ۳۸۴؛ سپس `LayerNorm(384) -> Linear(384,128) -> SiLU -> Dropout(0.10) -> Linear(128,1)`. خروجی یک logit برش است؛ sigmoid فقط در inference/validation. تمام backbone `requires_grad=False` و همیشه eval است، ازجمله BatchNorm؛ فقط سه projection و head trainable هستند. تنها mode فعلی GAP است؛ طراحی `_pool` امکان ablation جداگانهٔ آینده را بدون تغییر hook می‌دهد، ولی spatial top-k اکنون پیاده‌سازی نشده است.

**داده و جلوگیری از leakage:** همان `outputs/run_a_fracture_classifier_oof/slice_label_manifest.csv` و همان `ReviewedSlices`/`image_tensor` classifier P5 استفاده می‌شود؛ فایل‌های label و وزن منبع با SHA256ِ پروتکل P5 قبلی تطبیق داده می‌شوند. Reviewed slice با حداقل یک باکس معتبر مثبت؛ reviewed slice با annotation صریحِ بدون باکس منفی؛ ۲۴ برش missing/unreviewed JSON از BCE کنار گذاشته می‌شوند و **منفی فرض نمی‌شوند**. Foldهای بیمارمحور همان ۱۶۹ Study/۱۵۵ بیمار هستند؛ train/val patient overlap صفر. `pos_weight` داخل هر Fold فقط از چهار Fold train محاسبه می‌شود. شمارش‌ها:

| Held-out Fold | train/val Study | train/val patient | train مثبت/منفی slice | val مثبت/منفی slice | train-only pos_weight |
|---:|---|---|---|---|---:|
| 0 | 136/33 | 124/31 | 177/3341 | 46/854 | 18.875706214689266 |
| 1 | 136/33 | 124/31 | 177/3364 | 46/831 | 19.005649717514125 |
| 2 | 137/32 | 124/31 | 178/3371 | 45/824 | 18.938202247191010 |
| 3 | 133/36 | 124/31 | 180/3353 | 43/842 | 18.627777777777776 |
| 4 | 134/35 | 124/31 | 180/3351 | 43/844 | 18.616666666666667 |

**پروتکل فریز‌شده:** همان PNG review‌شدهٔ Run A از 2.5D ±5 mm فیزیکی، HU WL=800/WW=1600؛ resize/RGB/normalization دقیقاً تابع P5 قبلی، imgsz=768؛ inference DICOM نیز دقیقاً همان `make_hu_input` و `image_tensor` قبلی را reuse می‌کند و همهٔ SOPهای cache کامل Study منتخب را می‌گیرد. Mosaic، MixUp، window یا augmentation جدید، HNM، Stage 2 و تغییر backbone وجود ندارد. seed هر Fold=`42+fold` قبل از ساخت مدل؛ 10 epoch، batch8، AdamW LR=`1e-4` برای head، weight decay=`5e-4`، BCEWithLogitsLoss با train-only pos_weight. `best_classifier.pt` فقط با بیشینهٔ validation **slice PR-AUC** انتخاب می‌شود؛ tie epoch قبلی را نگه می‌دارد. این همان قانون آزمایش P5 است.

**فایل‌های جدید و جداسازی:** `src/fracture_classifier/feature_hooks.py`، `multiscale_model.py`، `multiscale_training.py`، `multiscale_inference.py`، `multiscale_compare.py` و `tests/test_multiscale_classifier.py`. هیچ فایل P5 موجود، `submit/`، Run A/B/FULL169 یا وزن تغییر نکرد. آموزش آینده فقط در `outputs/run_a_fracture_multiscale_fold0/` تا `fold4/` و held-out cache فقط در `outputs/run_a_fracture_multiscale_oof/` می‌نویسد. runner اگر پوشهٔ Fold از قبل موجود باشد overwrite را رد می‌کند. inference برای هر Study resumable است و SOP list/Study/patient/وزن/preprocessing/cache signature را تطبیق می‌دهد؛ GPU با کمتر از 8 GiB آزاد رد می‌شود. comparator تا تکمیل واقعی هر پنج Fold/cache و برابری دقیق ۱۶۹ Study جلو نمی‌رود.

**طرح مقایسهٔ بعد از آموزش، اجرا نشده:** study score اصلی classifier=`top5_mean`، ثابت از انتخاب هر پنج Fold P5 قبلی؛ `max`، `top3_mean` و `top10_percent_mean` فقط diagnostic خواهند بود و از pooled OOF انتخاب نمی‌شوند. پنج سیستم A آشکارساز Run A، B classifier P5، C classifier P3+P4+P5، D fusion قبلی P5، E fusion جدید با همان `0.75*RunA_OOF_detector_probability + 0.25*classifier_top5` سنجیده می‌شوند. classifier قدیمی/جدید و fusionها روی همان Study و همان detector fold-specific OOF paired هستند. metricهای fracture Study، oracle-other-heads triage Macro-F1، classwise F1، confusion matrix، FN rescue و FP جدید، و paired stratified patient-cluster bootstrap ۵۰۰۰تایی با seed 20260916 برای E در برابر A و D ذخیره خواهد شد. ICH/MLS واقعی فقط در ارزیابی oracle وارد می‌شود؛ این Macro-F1 نهایی submission نیست. میانگین slice PR-AUC قدیمی P5=`0.13509429`، foldهای ۰..۴ به‌ترتیب `.16135245/.10082845/.08150862/.24723180/.08455014` بود. criteria قبل از نتیجه ثبت شده: mean slice PR-AUC>=0.17، ترجیحاً ۴/۵ Fold بهتر، TP>=10، FN<=14، FP<=5، sensitivity>=.417، specificity>=.966، PR-AUC>=.56 و oracle Macro-F1>`.9696273781` بدون افت شدید classwise. بهترشدن slice PR-AUC به‌تنهایی GO نیست. اگر شکست بخورد Stage 2 خودکار آغاز نمی‌شود.

**تست/ممیزی واقعی، نه نتیجهٔ مدل:** اجرای `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m py_compile src/fracture_classifier/feature_hooks.py src/fracture_classifier/multiscale_model.py src/fracture_classifier/multiscale_training.py src/fracture_classifier/multiscale_inference.py src/fracture_classifier/multiscale_compare.py tests/test_multiscale_classifier.py` پاس شد. `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_multiscale_classifier.py` در اجرای نهایی برابر **8 passed in 5.35s** بود. تست‌ها گراف واقعی هر پنج وزن، forward مصنوعی ۷۶۸ CPU، shape/stride/channels، freeze، برابری data/پیش‌پردازش P5، شمار Foldها و pos_weight، رد unreviewed و patient leakage، seed، top5، bootstrap، رد cache ناقص و بازتولید pairing تاریخی Run A/P5 در ۱۶۹ Study را پوشش می‌دهند. `--help` هر سه CLI training/inference/compare موفق بود. `--audit` پنج Fold بدون ساخت output و بدون GPU اجرا شد. comparator جدید hash وزن P5 قبلی را با cache تاریخی تطبیق می‌دهد؛ inference نیز hash FULL169 محافظت‌شده را می‌سنجد. hashهای Run A پنج‌فولد و FULL169 با ثبت‌های قبلی یکسان بودند؛ `submit/models/best.pt` نیز همان SHA256 FULL169 یعنی `109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640` ماند.

**فرمان‌های دستیِ آینده، اکنون اجرا نشده‌اند:** از ریشهٔ repo و فقط پس از آزاد بودن GPU:

```bash
tmux new-session -d -s fracture_ms_fold0 'cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12 && PYTHONPATH=src .venv/bin/python -u -m fracture_classifier.multiscale_training --fold 0 --device cuda:0 --batch 8 > outputs/run_a_fracture_multiscale_fold0.log 2>&1'
tail -f outputs/run_a_fracture_multiscale_fold0.log
```

پس از پایان و بررسی Fold 0، Foldهای ۱ تا ۴ فقط به‌صورت ترتیبی:

```bash
tmux new-session -d -s fracture_ms_folds1to4 'cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12 && for FOLD in 1 2 3 4; do PYTHONPATH=src .venv/bin/python -u -m fracture_classifier.multiscale_training --fold "$FOLD" --device cuda:0 --batch 8 > "outputs/run_a_fracture_multiscale_fold${FOLD}.log" 2>&1 || exit 1; done'
```

بعد از `COMPLETE.json` هر پنج Fold:

```bash
tmux new-session -d -s fracture_ms_oof 'cd /mnt/mohammad.rezaei/TRAIN_SKULL/YOLO_12 && PYTHONPATH=src .venv/bin/python -u -m fracture_classifier.multiscale_inference --fold 0 --fold 1 --fold 2 --fold 3 --fold 4 --device cuda:0 --batch 8 > outputs/run_a_fracture_multiscale_oof.log 2>&1'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m fracture_classifier.multiscale_compare --bootstrap-repeats 5000 --seed 20260916
```

**محدودیت و وضعیت فعلی:** هیچ پوشهٔ `outputs/run_a_fracture_multiscale_fold*` یا `outputs/run_a_fracture_multiscale_oof/` هنوز ساخته نشده؛ بنابراین هیچ fold آموزش ندیده و هیچ metric جدیدی وجود ندارد. fixed test، FULL169، Run A، checkpointهای P5 و `submit/` دست‌نخورده‌اند. هیچ GPU inference/training یا FULL169 training شروع نشد؛ فقط graph inspection و synthetic forward CPU انجام شد. مرحلهٔ بعدی فقط اجرای دستی Fold 0 و قضاوت بر اساس validation آن، سپس تکمیل Foldهای دیگر و OOF است.

## آماده‌سازی کنترل نسخه برای انتشار کد — 2026-09-17

**درخواست و دامنه:** آماده‌سازی این checkout برای ارسال صرفاً کد به مخزن GitHub `MahsaNasehi/SkullNet` روی شاخهٔ `mohammad`، بدون checkpoint، داده، خروجی آموزش یا log حجیم.

**تغییر:** `.gitignore` با الگوهای محلی تکمیل شد: artifactهای وزن اضافی (`*.bin`، `*.h5` و `*.hdf5`)، cacheهای TensorBoard/Weights & Biases/MLflow و archiveهای `*.7z`/`*.rar` اکنون نادیده گرفته می‌شوند. الگوهای پیشین `outputs/`، `*.log`، `*.pt`/`*.pth`/`*.ckpt`، `weights/`، مسیرهای داده و `.venv/` همچنان پوشش اصلی فایل‌های حجیم این پروژه هستند.

**بررسی واقعی و محدودیت:** بازبینی فایل‌سیستم نشان داد checkpointهای تولیدشده زیر `outputs/**/weights/` تا حدود 79 MB برای هر فایل وجود دارند و همگی با قاعدهٔ `outputs/` از stage شدن خودکار خارج‌اند؛ محیط مجازی نیز با `.venv/` خارج است. با یک repository موقت و `git check-ignore --no-index`، نادیده‌گرفتن checkpoint، log، داده و `.venv` تأیید شد و یک فایل source نمونه قابل track ماند. metadata گیت موجود در این محیط خالی و فقط‌خواندنی است (`.git` فاقد `HEAD` و `config` است) و تلاش `git init -b mohammad` پیش از ایجاد metadata با خطای `Read-only file system` برای `.git/hooks/` متوقف شد؛ بنابراین این checkout هنوز مخزن Git قابل استفاده‌ای ندارد و نه branch، نه remote و نه push قابل اجرای محلی نیست. هیچ checkpoint، log یا داده‌ای حذف یا جابه‌جا نشد. پیش از push واقعی، یک repository سالم باید initialize/clone شود و بعد با `git check-ignore` و `git status` تأیید شود که فقط کد stage می‌شود.
