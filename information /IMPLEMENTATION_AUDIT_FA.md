# ممیزی پیاده‌سازی و برنامهٔ QWK بالاتر از ۰٫۹۹ برای Fracture

> **به‌روزرسانی Stage B:** پیاده‌سازی weak supervision در سطح Study برای ۱۴۰ مطالعهٔ Tier2، تست‌ها و دستورهای آزمایش در [WEAK_STUDY_REFINEMENT_REPORT_FA.md](WEAK_STUDY_REFINEMENT_REPORT_FA.md) آمده است.

**مسیر مبنا:** `/mnt/mohammad.rezaei/TRAIN_SKULL/SkullNet_yolo_large`  
**تاریخ ممیزی:** ۱۴۰۵/۰۶/۱۸ (2026-09-09)  
**دامنه:** فقط `fracture detection`. فایل Speaker Identification و توسعهٔ مدل‌های ICH/MLS خارج از دامنه‌اند؛ خروجی OOF آن دو head فقط در مرحلهٔ ارزیابی مشترک لازم است.

## نتیجهٔ اصلی

مدل فعلی `submit/best.pt` یک **YOLO11s سفارشی با head سطح P2** است، نه YOLO معمولی استاندارد و نه Large. ورودی آن CT به‌صورت **2.5D سه‌کاناله**، bone window با `WL=800, WW=1600` و اندازهٔ `768×768` است. headهای تشخیص strideهای `4, 8, 16, 32` دارند و مدل `9,574,772` پارامتر دارد. وزن submission دقیقاً همان بهترین checkpoint فولد ۱ است:

- SHA256: `17759933df8b783c6994baaddbb9758b848a2273e0280fc7c0adf0753922b0a8`
- run: `v5_hu800_ww1600_original`
- aggregation: `top3_mean`
- detector proposal floor: `0.01`

**بزرگ کردن مدل به‌تنهایی مسیر محتمل رسیدن به ۰٫۹۹ نیست.** اجرای فعلی `YOLO11l-P2` بعد از ۱۳ epoch متوقف شده و بهترین `mAP50-95=0.03987` آن از baseline همان فولد با `0.04701` پایین‌تر است. هنوز برای Large ارزیابی study-level یا QWK وجود ندارد، پس نتیجه قطعی دربارهٔ ظرفیت نمی‌توان داد؛ ولی شواهد فعلی هیچ منفعتی برای Large نشان نمی‌دهد. اجرای rotation در ۹ epoch به `mAP50-95=0.05255` رسیده و از baseline بهتر است، بنابراین تکمیل آزمایش تنوع منفی قبل از هزینه‌کرد روی Large اولویت بالاتری دارد.

عددهای معتبر پس از اجرای دوبارهٔ ارزیابی با تابع رسمی triage:

| ارزیابی روی همان ۵-fold OOF ثابت | QWK ایزوله | نکته |
|---|---:|---|
| `top3_mean`، threshold=0.5 | **0.969723** | 12 TP، 20 FP، 16 FN |
| همه‌منفی | **0.988725** | baseline ساده و بسیار قوی |
| بهترین threshold روی کل OOF | **0.988725** | exploratory و خوش‌بینانه؛ از ۰٫۹۹ عبور نمی‌کند |
| threshold به‌صورت fold-held-out | **0.983085** | برآورد سالم‌تر؛ ناپایداری threshold را نشان می‌دهد |
| fracture oracle | **1.000000** | فقط سقف نظری |

گزارش‌های قبلی `reports/oof_original*` پیش از اصلاح تابع triage ساخته شده بودند و نباید برای تصمیم جدید استفاده شوند. خروجی معتبر جدید در `reports/oof_original_official_spatial_v2/` قرار دارد.

## چرا هدف ۰٫۹۹ ممکن است، ولی تضمین‌شدنی نیست

فقط شش مطالعه در دادهٔ train وجود دارد که با oracle بودن ICH/MLS، تشخیص درست fracture کلاس triage را عوض می‌کند:

`1196, 271623, 271912, 272626, 3796, 4240`

baseline همه‌منفی QWK=`0.988725` دارد. درست کردن **فقط یکی** از این شش مطالعه، اگر false positive مؤثر تازه‌ای ایجاد نشود، QWK را به `0.990627` می‌رساند. بنابراین هدف عددی روی این داده از نظر ریاضی ممکن است، اما مسئلهٔ اصلی افزایش عمومی recall نیست؛ باید یکی از مطالعات تصمیم‌ساز بازیابی شود و specificity بسیار بالا بماند.

| سری | fold | max slice | top3_mean | برداشت |
|---|---:|---:|---:|---|
| 1196 | 2 | 0.966 | 0.899 | proposal خوب دارد؛ threshold/تصمیم مهم است |
| 271623 | 0 | 0.395 | 0.267 | evidence ضعیف تا متوسط |
| 271912 | 0 | 0.718 | 0.577 | proposal قابل استفاده دارد |
| 272626 | 1 | **0.000** | **0.000** | detector اصلاً proposal نداده؛ aggregation حلش نمی‌کند |
| 3796 | 0 | 0.502 | 0.255 | یک spike تک‌برشی |
| 4240 | 2 | 0.264 | 0.218 | evidence ضعیف |

برای `272626` تغییر threshold، calibration، spatial aggregation یا verifier هیچ کمکی نمی‌کند؛ باید proposal recovery با رزولوشن/crop یا branch مطالعه‌ای رخ دهد. برای بقیه، کنترل FP و post-processing می‌تواند مؤثر باشد.

## نتیجهٔ آزمایش Large و اندازهٔ مناسب مدل

طبق جدول رسمی Ultralytics، خانوادهٔ استاندارد YOLO11 از حدود 9.4M پارامتر برای `s` به 20.1M برای `m` و 25.3M برای `l` می‌رود. نسخه‌های این پروژه به‌دلیل P2 سفارشی دقیقاً همان شمارش استاندارد را ندارند؛ شمارش واقعی checkpointها ملاک است:

| مدل | پارامتر واقعی/وضعیت | بهترین mAP50-95 فولد ۰ | نتیجه |
|---|---:|---:|---|
| YOLO11s-P2 baseline | 9.575M، کامل | 0.04701 | مرجع |
| YOLO11s-P2 + rotation | 9 epoch، ناقص | **0.05255** | سیگنال مثبت اولیه؛ QWK هنوز نامعلوم |
| YOLO11l-P2 | 26.076M، 13 epoch، ناقص | 0.03987 | تا اینجا پایین‌تر از baseline |
| YOLO11m-P2 | اجرا نشده؛ وزن `yolo11m.pt` موجود نیست | — | مقایسهٔ ظرفیت پیشنهادی بعدی |

checkpoint Large سالم است، optimizer دارد و epoch ذخیره‌شدهٔ آن ۱۲ است. نسخهٔ path-rebased آن برای resume محلی در `outputs/fold_0/v8_hu800_l_p2/weights/last_local_paths.pt` ساخته شده است. با این حال، ادامهٔ Large باید بعد از نتیجهٔ study-level آزمایش rotation انجام شود. m-P2 برای تست ظرفیت انتخاب بهتری است، چون هزینه و overfit آن بین s و l قرار دارد.

## ممیزی چیزهایی که پیاده‌سازی شده بود

بخش‌های زیر درست یا نزدیک به درست بودند:

- split پنج‌فولدی بر اساس بیمار؛ ۳۳۸ study و ۳۲۰ بیمار.
- جداسازی ۱۷۵ slice با وضعیت `unknown` از supervision منفی detector.
- تبدیل DICOM به HU، مدیریت `MONOCHROME1`، مرتب‌سازی فیزیکی و context 2.5D بر حسب میلی‌متر.
- P2 head برای fractureهای کوچک، COCO initialization محلی و ثبت نسبت انتقال وزن.
- OOF prediction در سطح بیمار و ثبت hash وزن/config.
- sampler منفی چرخشی deterministic با سقف تعداد slice هر study؛ اجرای واقعی نشان می‌دهد در هر epoch هر ۲۶۹ study مخزن منفی پوشش داده شده‌اند.
- اسکلت inner split، ROI verifier و spatial aggregation.

مواردی که ناقص یا نادرست بودند و در این ممیزی اصلاح شدند:

1. **وابستگی دیتاست به مسیر اشتباه:** manifest، txt و YAMLهای دیتاست محلی هنوز `SkullNet-main` را نشان می‌دادند. همه به `SkullNet_yolo_large` rebase شدند؛ فایل‌های قبلی در `.path_rebase_backup` و hashها در `path_rebase_provenance.json` حفظ شده‌اند.
2. **دوپاره شدن run rotation:** callback قبل از Ultralytics پوشه را می‌ساخت و run واقعی به `hu800_rotation-2` می‌رفت. callback اکنون فقط پس از ساخته‌شدن trainer از `trainer.save_dir` استفاده می‌کند.
3. **خاموش شدن rotation در resume:** callback سفارشی داخل checkpoint serialize نمی‌شود. اکنون در resume نیز نصب می‌شود.
4. **از بین رفتن provenance اولیه در resume:** گزارش انتقال pretrained دیگر overwrite نمی‌شود؛ resume event جداگانه ثبت می‌شود.
5. **تشخیص run ناقص:** completion marker اضافه شد و script در صورت وجود `last.pt` معتبر resume می‌کند؛ وجود صرف `best.pt` دیگر به معنی پایان آموزش نیست.
6. **Platt calibration پیش‌فرض:** artifact قبلی پس از Platt، TP=0 داشت. پیش‌فرض OOF به `none` تغییر کرد و calibration فقط در صورت اثبات cross-fitted پذیرفته می‌شود.
7. **spatial tracking ضعیف:** نسخهٔ قبلی فقط قوی‌ترین box هر slice را می‌دید، مختصات pixel خام داشت و track را با بیشترین تک-score انتخاب می‌کرد. نسخهٔ جدید همهٔ proposalها، اندازهٔ تصویر، مجاورت فیزیکی و یک مسیر مستقل برای fracture تک‌برشی را استفاده می‌کند.
8. **نبود geometry در OOF قدیمی:** `generate_oof` اکنون با `--fold-slices` boxهای CSVهای تاریخی را به studyها وصل می‌کند.
9. **نبود threshold protocol:** ابزار `threshold_selection.py` اضافه شد و threshold را fold-held-out گزارش می‌کند.
10. **double weighting در verifier:** هم balanced sampler و هم `pos_weight` هم‌زمان مثبت‌ها را دو بار وزن می‌دادند. اکنون balanced sampler با BCE بدون وزن استفاده می‌شود.
11. **guardهای verifier:** inner fold اعلام‌شده، protocol، hash checkpoint، duplicate slice، دوکلاسه بودن validation و برابری preprocessing train/inference کنترل می‌شوند.
12. **HNM خوش‌بینانه:** global OOF HNM فقط برای ساخت صف بررسی مجاز است. hard negative آموزشی باید `inner_oof`، متعلق به outer fold مربوط و با `review_status=confirmed_negative` باشد.
13. **اسکریپت‌ها و configها:** مسیر پیش‌فرض داده و Python به محیط همین پروژه اصلاح شد؛ OOF geometry و calibration none به اسکریپت‌ها اضافه شد.

## چیزهایی که هنوز تکمیل نشده‌اند

- rotation کامل نشده و study-level evaluation ندارد.
- Large کامل نشده و QWK ندارد.
- وزن pretrained مدل m موجود نیست و m اجرا نشده است.
- `reannotation_log.csv` فقط header دارد؛ هیچ candidate با وضعیت نهایی review نشده است.
- inner-fold detector predictionها و verifier weights وجود ندارند.
- spatial model روی OOF تاریخی بهتر نشد: PR-AUC آن `0.38681` در برابر `0.41708` برای top3 بود؛ بنابراین deploy نشد.
- StudyMIL در submission فعلی حضور ندارد.
- خروجی OOF مدل‌های واقعی ICH و MLS تیم موجود نیست؛ QWK مشترک و decision-aware gating هنوز قابل اعتبارسنجی نیست.
- `submit/model.py` پنج حجم و MLS را صفر می‌گذارد. این برای تست fracture-only قابل فهم است، ولی QWK آن معادل QWK نهایی تیم یا ارزیابی isolated با oracle headهای دیگر نیست.

## آزمایش تصمیم‌محور با بیشترین شانس عبور از ۰٫۹۹

قانون رسمی fracture را فقط به‌صورت یک bit در threshold `0.5` می‌خواند و اثر آن به حجم ICH وابسته است. به همین دلیل خروجی fracture باید در ارزیابی مشترک با head حجم بررسی شود. تحلیل counterfactual روی OOF فعلی، با استفاده از **حجم GT فقط برای اثبات ایده**، چنین نتیجه‌ای داد:

| gate حجمی oracle | threshold بهینهٔ exploratory | QWK |
|---|---:|---:|
| بدون gate | 0.9285 | 0.988725 |
| `15 <= total_volume < 60 mL` | 0.2063 | **0.992628** |

این عدد قابل submission نیست، چون از GT volume استفاده کرده است. معنی آن این است که با رسیدن OOF حجم هم‌تیمی‌ها، باید یک **joint cross-fitted decision layer** کوچک آزمایش شود:

1. ورودی‌ها: fracture raw score، total predicted ICH volume، MLS predicted و فاصله تا مرزهای ۱۵/۴۰/۶۰ mL و ۳/۵ mm.
2. روی outer-train یا inner folds threshold/gate انتخاب شود.
3. روی outer held-out همان fold بدون تنظیم دوباره ارزیابی شود.
4. با سه حالت مقایسه شود: fracture=0، fracture مستقل، fracture تصمیم‌محور.
5. فقط اگر QWK pooled و bootstrap بیمار بهتر شد، وارد submission تیم شود.

این مسیر مستقیماً objective مسابقه را بهینه می‌کند و در دادهٔ موجود شواهد عددی قوی‌تری از تعویض s با l دارد.

## ترتیب اجرای پیشنهادی

### ۱. آزمایش کنترل‌شدهٔ rotation

اجرای قدیمی rotation را resume نکنید؛ نام run و project داخل checkpoint قدیمی ناسازگار است و فقط ۹ epoch هزینه شده است. run اصلاح‌شده را با نام تازه اجرا کنید:

```bash
cd /mnt/mohammad.rezaei/TRAIN_SKULL/SkullNet_yolo_large
PYTHONPATH=src scripts/run_improved_pipeline.sh train-rotation-fold0
```

Gate عبور: در فولد ۰ علاوه بر mAP، held-out PR-AUC، sensitivity، specificity، QWK و شش مطالعهٔ تصمیم‌ساز بررسی شوند. اگر rotation در study-level بهتر نبود، اجرای پنج فولد آن متوقف شود. اگر بهتر بود، هر پنج fold اجرا و OOF مستقل ساخته شود.

### ۲. بازبینی خطاها

ابتدا همهٔ FPهای مؤثر و شش مطالعهٔ تصمیم‌ساز review شوند. برای هر FP علت یکی از این‌ها ثبت شود: suture، vascular channel، skull base، motion/metal، partial volume، annotation error. verifier یا HNM پیش از `confirmed_negative` شدن نمونه نباید آموزش ببیند. مقالهٔ جدید depressed-skull-fracture نیز vertex و partial-volume/slice-thickness را از منابع خطای مهم گزارش کرده است؛ این taxonomy با دادهٔ خود پروژه باید تأیید شود.

### ۳. بازیابی proposalهای از دست‌رفته

برای `272626` و هر FN با max≈0، یک آزمایش واحد انجام شود:

- ابتدا crop جمجمه و `imgsz=1024` روی s-P2؛ یا
- branch سبک study MIL/attention MIL که کل سری را می‌بیند؛ یا
- context عمیق‌تر 5/7-slice یا feature-level 3D context.

هر بار فقط یک متغیر تغییر کند. 3DCE و پژوهش‌های مشابه نشان می‌دهند feature-level 3D context برای تمایز lesion از ساختار مشابه در CT منطقی است؛ این پشتوانهٔ روش است، نه تضمین نتیجه روی این دیتاست کوچک.

### ۴. verifier و HNM فقط با nested protocol

برای هر outer fold، داخل outer-train سه inner fold بسازید. detector inner-OOF proposal تولید کند؛ منفی‌های دشوار توسط انسان تأیید شوند؛ verifier/HNM فقط از همان outer-train استفاده کند؛ سپس outer-val یک بار ارزیابی شود. threshold و epoch verifier نیز داخل inner protocol انتخاب شوند.

### ۵. ظرفیت

پس از rotation، `m-P2` را روی فولد ۰ اجرا کنید. اگر m در PR-AUC/QWK و نوع خطای مکمل بهتر شد، پنج fold آن ارزش دارد. Large موجود را فقط برای تکمیل evidence می‌توان resume کرد:

```bash
PYTHONPATH=src scripts/run_improved_pipeline.sh resume-l-fold0
```

این دستور از checkpoint path-rebased محلی استفاده می‌کند. صرف بهتر شدن train loss یا detector recall دلیل deploy نیست.

### ۶. ترکیب با headهای تیم

پس از تحویل OOFهای ICH/MLS با همان patient folds، آزمایش gate تصمیم‌محور اجرا شود. این مرحله در حال حاضر تنها آزمایشی است که روی پیش‌بینی‌های موجود شواهد counterfactual بالاتر از ۰٫۹۹ دارد.

### ۷. پذیرش نهایی

- هر ۳۳۸ study دقیقاً یک OOF prediction و هیچ patient overlap نداشته باشد.
- threshold، calibration، HNM، verifier و انتخاب epoch روی همان outer held-out تنظیم نشده باشند.
- QWK رسمی مشترک با خروجی واقعی سه head گزارش شود؛ fracture-only score جدا بماند.
- confusion matrix شکستگی، PR-AUC، sensitivity، specificity و bootstrap بیمار ثبت شود.
- مدل نهایی در GPU با سقف حافظهٔ محیط رسمی و بودجهٔ زمانی کل سه head benchmark شود.
- `submit/best.pt` یا deployment manifest فقط بعد از این ارزیابی عوض شود.

## وضعیت تست و artifactها

- پس از اضافه‌شدن Stage B، کل suite برابر `102 passed` است؛ هشت تست جدید مستقیماً loss routing، gradient، leakage، Tier2 exclusion و pooling را پوشش می‌دهند.
- audit خام: ۳۳۸ study، ۷۶۸۳ DICOM، ۲۸ مطالعهٔ مثبت، ۲۶۰ slice مثبت، ۳۵۶ box، ۱۷۵ unknown و صفر خطای decode/annotation.
- dataset path rebase: ۷۶۸۳ ردیف؛ semantic content بدون تغییر؛ hash قبل و بعد ثبت شده است.
- submission فعلی عمداً تغییر نکرد، چون هیچ artifact جدید هنوز OOF بهتر و معتبر نشان نداده است.

## منابع روش‌شناسی

- [مستندات رسمی YOLO11 و جدول اندازه/سرعت/دقت](https://docs.ultralytics.com/models/yolo11/)
- [3D Context Enhanced R-CNN برای lesion detection در CT](https://arxiv.org/abs/1806.09648)
- [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html)
- [مطالعهٔ تشخیص depressed skull fracture و تحلیل خطاهای vertex/partial volume](https://pmc.ncbi.nlm.nih.gov/articles/PMC13483734/)

این منابع ثابت نمی‌کنند Large یا یک معماری مشخص روی دادهٔ شما بهتر می‌شود؛ تصمیم نهایی باید با patient-level OOF همین پروژه گرفته شود.
