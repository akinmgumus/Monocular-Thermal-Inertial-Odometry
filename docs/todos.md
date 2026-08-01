# Proje Yapılacaklar & Açık Kararlar

Termal-ataletsel odometri (TIO) tezi için çalışma takibi.
`[x]` = bitti, `[ ]` = bekliyor, **KARAR GEREKLİ** = henüz netleşmedi.

---

## 1. Açık kararlar (uygulamadan önce senin onayın gerekli)

- [ ] **Dataset başına rapor figür yerleşimi** — *KARAR GEREKLİ.*
  - Dataset başına 4 konfigürasyon: {KLT, ORB} × {CLAHE açık, CLAHE kapalı}.
  - Her konfigürasyonun 6 grafiği: **X, Y, Z, XY, ATE, track count**.
  - Soru: hepsini **dataset başına tek birleşik figüre** mi sıkıştıralım, yoksa böl?
    - **A)** Tek büyük grid (4 config × 6 panel = 24 panel) → kompakt ama okunamayacak
          kadar küçük olabilir.
    - **B)** İki figür: 4'lü *trajectory* figürü (X/Y/Z/XY) + 4'lü *diagnostics*
          figürü (ATE + track count). ← muhtemelen en okunaklı seçenek.
    - **C)** Şu anki gibi konfigürasyon başına `_traj.png` + `_diag.png` (daha çok figür).
  - Eğilim: **B**. Birleşik plotter'ı yazmadan önce onayla.
- [ ] **Hangi ön-işleme sırası raporlanacak** (bkz. Bölüm 2): A/B koşularından sonra
        CLAHE→Norm ile Norm→CLAHE'den metriklere göre iyi olanı seç;
        **teze yalnız kazananın figürleri girecek**, kaybeden CSV'ye kaydedilecek
        ama raporda çizdirilmeyecek.
- [ ] **RPE boxplot stili** — A) koşu başına çok-pencereli (1/2/4/8 s) her dataset
        alt-bölümünde, B) dataset-arası (dataset/yöntem başına bir kutu) Tartışma için.
        Eğilim: **A + B**. Koşu başına tam RPE serisinin saklanması gerekir.
- [ ] **`all_runs.csv` temizliği** — şu an yalnızca append yapılıyor ve eski koddan
        kalma bozuk satırlar birikti (imkânsız `survival>duration`, `align_scale=0`).
        Seç: **A)** sıfırla + temiz re-run, **B)** `run_timestamp`/`run_id` kolonu +
        konfigürasyon başına en güncel satırı tut, **C)** dokunma.
        Eğilim: **A** (Bölüm 2 için zaten her şeyi baştan koşturacağız).
- [ ] **Termal feature-matching benchmark** (TO vs VO, IMU'dan bağımsız) — yapalım mı?
        Termal vs RGB'de KLT/ORB track ömrü/sayısı/inlier oranı. "Pipeline doğru,
        sınırlayıcı modalite" argümanını güçlendirir. Kapsamı belirle.
- [ ] **SThereo** — açık bir *negatif sonuç* olarak dahil edilsin mi (ileri hareket +
        yüksek hız + düşük kontrast)? Eğilim **evet**.
- [ ] **VOXL/AerialTN** — hiç dahil edilsin mi? Eksen-konvansiyonu yanıtına bağlı
        (Bölüm 7). Yedek plan: 1 paragraf "iskele hazır, engellendi" + ek (appendix).
- [ ] **Contributions bölümü** — şu an `03_Problem_Formulation` içinde; ayrıca
        Conclusion'da da özetlensin mi (danışman tercihi)?

---

## 2. Ön-işleme A/B deneyi — CLAHE↔Normalization sırası (RQ3)

Amaç: **CLAHE ile 8-bit normalizasyonun sırasının** doğruluğa etkisini ölçmek,
iyi olan sırayı seçmek ve raporda gerekçelendirmek.

- İki pipeline (denoise hep önce, gerisi aynı):
  - `dataloader.py`  → undistort → denoise → **CLAHE@16 → norm@8** (mevcut).
  - `dataloader2.py` → undistort → denoise → **norm@8 → CLAHE@8** (literatürde
        standart; CLAHE yoğun 256-bin histogramda çalışır). [BİTTİ: dosya oluşturuldu]
- [ ] **Her 16-bit termal datasette iki sırayı da koştur** (FIReStereo, ROVTIO,
        SThereo, VOXL). Değişim test scriptinde tek satırlık import
        (`dataloader` → `dataloader2`).
  - Not: **EuRoC 8-bit → sıra fark etmez**, orada A/B'yi atla.
- [ ] **İki sıra için ATE ve RPE'yi kaydet**; değişimi not et (Δ ATE, Δ RPE).
        **Tüm** sonuçları sakla (figürler + CSV) — her koşuya ayrı `out_base` son-eki
        ver (ör. `_claheNorm` vs `_normClahe`) ki hiçbir şey üzerine yazılmasın.
- [ ] Adil olması için, `clip_limit` 16-bit ile 8-bit'te farklı davrandığından,
        8-bit yolda CLAHE clip'ini süpürmek isteğe bağlı (1.5 / 2 / 3).
- [ ] **Metriklere göre iyi olan sırayı seç**; final rapor figürlerinde onu kullan.
        Teze yalnız kazananın grafikleri girer.
- [ ] **Birkaç cümle yaz** (Method/Results): iki sıra da test edildi, ATE/RPE
        karşılaştırıldı ve performansı daha iyi olan ön-işleme sırası seçildi
        (hangisi ve ne kadar farkla olduğunu belirt).

**Bu sıra kararı hangi bölümleri etkiler?**
- `05_Method` (ön-işleme alt-bölümü): seçilen sıra + gerekçesi.
- `07_Results`: A/B karşılaştırma sayıları (Δ ATE / Δ RPE).
- `08_Discussion`: RQ3 (ön-işleme hassasiyeti) yorumu.
- **Intro / Related Works / Problem Formulation'ı ETKİLEMEZ** — RQ3 zaten bu soruyu
  soruyor (genel "normalizasyon/ön-işleme hassasiyeti"), spesifik sıra adı geçmiyor.
  Yani bu üç chapter sıra kararını beklemeden bitirilebilir.

---

## 3. ZUPT — dataset başına sabit, matris faktörü DEĞİL

Karar: ZUPT'u test matrisine **ekleme** (kombinatoryal patlama + düşük araştırma
değeri). Dataset başına sabitle, gerekçesini yaz. (İKİNCİ GEÇİŞTE eklenecek.)

- [ ] **Method §ZUPT'u yeniden çerçevele:** ZUPT tüm datasetlerde değil, sadece
        **uzun statik/hover başlangıcı olan** platformlarda etkisini görmek için
        etkinleştirildi. Böylece okuyucu ZUPT'u her datasette beklemez. (ZUPT hover
        yapan hava araçlarında prensipte mantıklı: hover'da hız ~0.)
- [ ] **Trade-off bulgusunu ekle (§ZUPT gerekçesi + Discussion):** ZUPT statik fazda
        bias/hız drift'ini sınırlar; AMA rest-detektörü **yavaş gerçek hareketi
        'statik' sanıp v=0 zorlayarak o hareketi bozabilir** → bu yüzden yalnız
        uzun statik + gerisi yeterince dinamik olan datasetlerde açık.
- [ ] Küçük **dataset başına ZUPT tablosu** (dataset → açık/kapalı → sebep) + final
        koşulardaki gerçek flag'leri `test_msckf_vio_*.py`'den doğrula.
- [ ] (opsiyonel) **Tek ablation**: statik başlangıçlı ROVTIO'da ZUPT on/off —
        seçimi ampirik gerekçelendirir (tam faktör değil).

---

## 3b. Normalizasyon (percentile) hassasiyeti — RQ3 kanıtı (İKİNCİ GEÇİŞ)

- [ ] **Percentile-bound hassasiyet mini-deneyi:** ROVTIO'da ATE @ (1-99 / 2-98 /
        5-95), sabit method+CLAHE. Küçük bir tablo → RQ3 (preprocessing hassasiyeti)
        için doğrudan kanıt. Sonuçların bu düğmeye çok duyarlı olması **bulgudur.**
- [ ] **Bütünlük:** ana sonuçlar için **tek** percentile ayarını dondur (ör. 2-98);
        koşu-başına en iyi ayarı seçme (cherry-picking olur). Hassasiyeti ayrı göster.
- [ ] **Sebep notu:** ROVTIO'da preprocessing sonrası parlaklık dalgalanması →
        (a) sensör AGC (Tau2) + (b) per-frame percentile normalizasyon; ikisi de
        brightness-constancy'yi bozar. Ayırt için percentile vs fixed koştur.
        → objektif AGC notu **Datasets (ROVTIO)**; analiz **Results Failure-Mode.**

---

## 4. Filtre & front-end parametrelerini raporda yazıyla açıkla

Tüm tunable parametreler bir **parametre tablosu + prose açıklamasıyla** Method
bölümünde anlatılacak: her parametrenin ne işe yaradığı, neden o değerin seçildiği.

- [ ] **MSCKF / filtre parametreleri:**
  - `chi2_alpha` — per-track Mahalanobis (chi²) outlier kapısının güven seviyesi;
        gözlem reddini nasıl etkilediği.
  - `gn_max_iter` — ters-derinlik üçgenlemesi için Gauss-Newton iterasyon sayısı.
  - `min_parallax_deg` — bir feature'ın üçgenlenebilmesi için gereken minimum parallax.
  - `min_depth` / `max_depth` — kabul edilen landmark derinlik aralığı.
  - `pixel_noise_std` — görüntü ölçüm gürültüsü (termal vs RGB farkı).
  - `MAX_WINDOW` — sliding window'daki kamera poz sayısı.
  - `MAX_TRACKS_PER_UPDATE` — update başına işlenen maksimum track.
  - `init_*_std` — başlangıç kovaryansları (att/bg/vel/ba/pos).
  - IMU gürültü parametreleri (`Q_matrix`: noise density + bias random walk) ve
        provenance'ı (config dosyalarındaki türetim).
- [ ] **ORB parametreleri:** `n_features`, `grid_rows/cols`, `ratio_thresh`,
        `ransac_thresh`, `fast_threshold`, `edge_threshold`, `max_pixel_displacement`.
- [ ] **KLT parametreleri:** `n_features`, `fb_eps` (forward-backward toleransı),
        `lk_win_size`, `lk_max_level`, `quality_level`, `min_distance`,
        `ransac_thresh`, `max_pixel_displacement`.
- [ ] Her datasetin final değerlerini bir **tuning tablosunda** topla (datasete göre
        değişenleri vurgula); detaylı/alternatif değerler appendix'e.

---

## 5. Değerlendirme & çıktı (kod)

- [x] **TUM trajectory export** — 5 test scripti artık 8-kolon kaydediyor
        (`t x y z qx qy qz qw`); quaternion `msckf.nominal_rot.as_quat()`. align/plot
        8-kolonu sorunsuz kabul ediyor (doğrulandı).
- [x] **evo kurulumu + wrapper** — `evo v1.36.5` kuruldu; `scripts/evo_eval.py`
        (Python API: associate → SE(3) align → APE/RPE) bizim metriklerle karşılaştırıyor.
- [x] **ATE çapraz-doğrulaması** — FIReStereo'da **evo 13.249 m vs bizim 13.222 m**
        (%0.2 fark, sadece örnekleme). Bizim ATE implementasyonu evo ile tutarlı. ✓
- [x] **RPE evo'dan** — RPE alignment'tan BAĞIMSIZ (göreli poz, global hizalama
        sadeleşir), o yüzden evo'nun standart RPE'sini otorite kabul ediyoruz
        (per-metre). posyaw/SE(3) RPE'yi etkilemez.
- [x] **evo CSV toplama + tablo** — `evo_eval.py` her koşunun evo ATE/RPE'sini
        `results/evo_runs.csv`'ye yazıyor (run adına göre dedup). `evo_eval.py table`
        markdown + LaTeX booktabs tablo basıyor. SThereo KLT on/off girildi.
- [x] **Metrik bölüşümü netleşti** — **ATE = posyaw (4-DOF, bizim)**,
        **RPE = evo** (Grupp_2017), **SE(3) raporda YOK** (yalnız `evo_eval`'de
        `[validation]` satırı: ATE makinemizi evo'ya karşı doğrular, rapora girmez).
        `evo_runs.csv` kolonları: `ate_rmse_posyaw`, `rpe_rmse_evo_per_m`.
- [ ] **Matrisi doldur** — her koşuyu `evo_eval.py <traj> <dataset>` ile değerlendir
        (CSV birikir), sonra `evo_eval.py table` ile cross-dataset tabloyu üret.
- [ ] **RPE boxplot** — Bölüm 1'deki seçilen stil(ler)i uygula; RPE çeviri serisini
        `metrics` içinde sakla (şu an atılıyor).
- [ ] **Dataset-arası özet tablosu** — `results/all_runs.csv`'yi okuyup LaTeX
        `booktabs` tablosu üreten jeneratör (dataset/yöntem/config başına ATE/RPE/drift).
- [ ] **Birleşik dataset-başına plotter** — Bölüm 1 yerleşimi karara bağlanınca
        (4 config × 6 panel) `evaluation.py` içinde fonksiyonu yaz.
- [ ] **(isteğe bağlı) NIS chi-square tutarlılık grafikleri** — appendix için
        update-başına NIS vs %95 bandı (şu an yalnız kümülatif `used/in` oranı var).

---

## 6. Koşulacak deneyler (`results/` doldur)

Dataset başına config'i dondur, koştur, figür + CSV topla. Her dataset hem
KLT/ORB × CLAHE± matrisini hem de CLAHE↔Norm sıra A/B'sini koşar (Bölüm 2).

- [ ] EuRoC MH_03 — KLT ve ORB (8-bit sanity baseline; sıra A/B yok)
- [ ] FIReStereo frick_1 — final tuned koşu + sıra A/B
- [ ] ROVTIO alt1 — final tuned koşu + sıra A/B (en iyi çalışan termal vaka)
- [ ] SThereo valley_evening — negatif-sonuç koşusu + sıra A/B
- [ ] VOXL/AerialTN — engellendi (Bölüm 7)
- [ ] Her koşunun `results/all_runs.csv`'ye eklendiğini ve kararlaştırılan figürleri
      ürettiğini doğrula.

---

## 7. Tanı / açık araştırmalar

- [x] **ROVTIO divergence karakterize edildi** — GT'ye karşı ham hatada 3 rejim:
      t=0–6 s sıkı (<0.5 m); t≈7–118 s şekil takip edilir ama bounded ~5–8 m offset;
      **t≈118 s → katastrofik diverge** (parabolik x∼t², IMU dead-reckoning).
      Split figürler `scripts/plot_rovtio_split.py` (seg1 ATE 2.89 m, seg2 422 m).
- [ ] **ROVTIO neden ~118 s'de diverge ediyor?** O andaki track count / NIS'e bak
      (manevra? parallax/track çöküşü?) → rapora gerçek bir failure-mode açıklaması.
      Muhtemelen tanı serisini loglayan bir re-run gerekir.
- [ ] **VIO update-path divergence (genel)** — asıl açık bug: IMU-only vs VIO
      katastrofik patlama. Bug avına devam (proje hafızasına bakınız).

---

## 8. Engellenen / dış bağımlılık

- [ ] **VOXL `/imu_apps` eksen konvansiyonu** — Jonathan'ın yanıtı bekleniyor.
      accel FLU okuyor (z=+9.7) ama gyro y/z işaretleri FRD gibi; hiçbir flip kombinasyonu
      düzeltmedi. `voxl-imu-server`'ın uyguladığı tam raw→body dönüşümü lazım.
- [ ] **VOXL Allan kaydı** — isteğe bağlı; `kalibr_imu_chain.yaml` (şişirilmiş) kullanımda.

---

## 9. Rapor yazımı (Overleaf, DTU template)

Bölümler `Chapters/NN_*.tex`; taslaklar `docs/thesis/` içinde sahnelendi.
Kural: paragraf başına tek kaynak satırı (elle ~76-karakter kırma yok).

### 9a. Bitmek üzere olan chapter'lar — KALAN EKSİKLER

İçerik taslakları yazıldı; bitirmek için aşağıdaki küçük işler kaldı. Bunların
**hiçbiri deney koşusuna veya CLAHE↔Norm sıra kararına bağlı değil** — şimdi bitirilebilir.

- [x] `01_Introduction` — Motivation + Aim + Outline yazıldı. Kalanlar:
  - [ ] Dosya başındaki **eski yorum bloğunu temizle** (artık geçersiz "CITATION
        STATUS / required missing" notları — o entry'ler bib'e eklendi).
  - [ ] **Thesis Outline'ı final chapter setiyle eşle** — şu an `ch:results_and_discussion`
        (birleşik) diyor olabilir; ama plan ayrı `07_Results` + `08_Discussion`.
        Outline'daki bölüm adlarını/etiketlerini gerçek chapter'lara göre düzelt.
  - [ ] **"five datasets" ifadesini VOXL kararına göre dondur** (5 mi 4 mü — Bölüm 7/1).
- [x] `02_Related_Works` — taksonomi + uzaktan-yakına bölümler; "Positioning" gap'le
      bitiyor + ileri işaret. Kalanlar:
  - [ ] **Derleme kontrolü**: tüm `\cite` ve `\ref` çözünüyor mu (özellikle
        `ch:problem_formulation` ileri-referansı).
  - [ ] **UK yazım + TIO/VIO terminoloji** son okuması.
- [x] `03_Problem_Formulation` — Problem Statement, şemsiye RQ + RQ1-3, kalibrasyon
      parametresi→hata listesi, Contributions, Scope. Kalanlar:
  - [ ] **Dataset adı/sayısını VOXL kararına göre güncelle** (Problem Statement +
        Contributions + Scope'taki "five datasets" ve dataset listesi).
  - [ ] Contributions'taki sonuç-bağımlı iddiaların (ör. "viable only within
        favourable regimes") Results ile destekleneceğini not et (yazımda dikkat).
- [x] `bibliography.bib` — Mourikis_2007, Burri_2016, Vidas_2013, Saputra_2020,
      Li_Mourikis_2013, Sturm_2012, Zhang_2018, Grupp_2017.

### 9b. Genel / entegrasyon (üç chapter'ı da ilgilendiren)

- [ ] **Overleaf entegrasyonu** — `docs/thesis/` taslaklarını gerçek `Chapters/`'a
      kopyala; master `\include` sırasını güncelle (`03_Problem_Formulation` eklendi,
      eski `03_Objectives` kaldırıldı). Sıra: Intro → RW → Problem Formulation → ...
- [ ] **Abstract** — henüz yazılmadı (en son yazılır ama eksik kalem).
- [ ] **Dataset sayısı (5 vs 4)** — VOXL kararı verilince Intro/Problem Formulation/
      Contributions'taki tüm "five datasets" geçişlerini tek seferde güncelle.
- [ ] **posyaw düzeltmesini tez metnine yansıt** — şu an üç yerde "SE(3) doğru
      hizalama" yazıyor, **posyaw / 4-DOF** olmalı (SE(3) roll/pitch'i, Sim(3)
      ölçeği gizler): (1) Related Works `sec:rw_eval`, (2) Problem Formulation
      Contributions §3 ("justified evaluation methodology"), (3) Background
      `sec:bg_metrics`. Zhang & Scaramuzza 2018'e dayandır.

### 9c. Yazılacak chapter'lar (içerik bekliyor)

- [ ] `04_Background` (IMU modeli, MSCKF matematiği, KLT/ORB, metrikler, SE(3) vs Sim(3))
- [ ] `05_Method` (mimari, ön-işleme + **CLAHE↔Norm seçimi**, front-end, back-end,
      init, **ZUPT gerekçesi**, **filtre/front-end parametre açıklamaları (Bölüm 4)**,
      değerlendirme çerçevesi)
- [ ] `06_Datasets` (sensör-takımı tablosu + 5 dataset + dataset başına ZUPT tablosu)
- [ ] `07_Results` (dataset başına figürler + dataset-arası tablo; yalnız kazanan
      ön-işleme sırasının grafikleri)
- [ ] `08_Discussion` (rejim bağımlılığı, ön-işleme/kalibrasyon hassasiyeti, ölçek
      gözlemlenebilirliği, failure mode'lar dahil ROVTIO 118 s, sınırlamalar)
- [ ] `09_Conclusion` (RQ1-3 yanıtları, gelecek çalışma)
- [ ] `Appendix` (tuning tabloları, NIS grafikleri, IMU-gürültü provenance'ı, VOXL
      konvansiyon araştırması, bug-avı günlüğü)

---

## 10. Kararlaştırıldı (referans — tekrar tartışma yok)

- **posyaw (4-DOF) hizalama** — VIO'da roll/pitch (yerçekimi) ve ölçek (ivmeölçer)
  gözlemlenebilir; yalnız global pozisyon + yaw serbest (Zhang & Scaramuzza 2018).
  SE(3) (6-DOF) roll/pitch hatasını, Sim(3) ölçek hatasını **gizler**. Varsayılan
  `align_mode='posyaw'` (`evaluation.py`). Eski karar SE(3)'tü → **düzeltildi**.
  Doğrulama: 5° roll hatasında SE(3) ATE=0, posyaw ATE=0.18 m (sentetik); SThereo
  KLT-off'ta SE(3) 76.3 vs posyaw 80.3 m (gerçek attitude drift'i açığa çıkardı).
- **Metrik bölüşümü**: **ATE = posyaw** (4-DOF, bizim implementasyon); **RPE = evo**
  (standart, alignment-bağımsız); **SE(3) raporda yer almaz** (yalnız geliştirme
  doğrulaması). evo posyaw'ı desteklemediği için headline ATE bizden gelir.
- **İki-figür çıktı**: `_traj.png` (X/Y/Z + XY) ve `_diag.png` (ATE + track count).
  Origin-anchored, **posyaw-aligned**. Eksenler GT'yi tam gösterir (otomatik kırpma
  yok; diverge eden VIO'nun kareden çıkmasına izin verilir).
- **Terminoloji**: bizim sistem = **TIO**; VIO genel paradigma ve RGB (EuRoC)
  baseline için saklı.
- **Tez yapısı**: Intro (Motivation+Aim) → Related Works → Problem Formulation →
  Background → Method → Datasets → Results → Discussion → Conclusion.
- **Survival time**: hesaplanıyor ama headline metrik DEĞİL.
- **Rolling-shutter telafisi**: uygulanmadı; sınırlama olarak raporlanacak.
- **Yalnız monocular** (sol kamera); stereo gelecek çalışma.
- **Filter-based** back-end MSCKF'e sabit; optimizasyon-tabanlı yalnız incelendi.
- **Deployment vs değerlendirme**: konuşlanan sistem *ham* metrik trajectory verir;
  hizalama yalnız GT ile karşılaştırma için var.
- **CLAHE'den önce denoise** (prensipli: sensör speckle'ını yanlış köşeye çevirmesin).
  Yalnız CLAHE↔Norm sırası test ediliyor (Bölüm 2).
- **`chi2_alpha` metadata düzeltmesi** [BİTTİ]: MSCKF artık `self.chi2_alpha` saklıyor;
  tüm `test_msckf_vio_*.py` hardcoded değer yerine `msckf.chi2_alpha` yazıyor.
- **ROVTIO GT** mevcut ve geçerli (Vicon, 12102 sample); eski "GT yok" notu güncel değildi.
## FIGÜR PLANI (Results) — 2026-07-08
Hedef ~12-15 figür (her run'ın 3'lüsü DEĞİL; temsili run + tablo).
- [ ] Preprocess: orijinal vs preprocess-sonrası termal (1-2 görsel).
- [ ] Aynı frame'in Norm→CLAHE vs CLAHE→Norm sonucu — 3 datasetten 1'er (tek multi-panel).
- [ ] (opsiyonel) histogram analizi — asıl önemli olan trajectory katkısı; emin değil, atlanabilir.
- [ ] Feature-match: ardışık 2 frame, preprocess+FE sonrası eşleşmeler — 3 örnek (1 bile yeterli olabilir).
- [ ] ROVTIO parlaklık-sabitliği ihlali: preprocess-sonrası ardışık 4 frame — percentile norm'a rağmen ani parlaklık değişimi → brightness-constancy ihlali → VO katkısına zarar önermesi.
- [ ] Per-dataset: 1 temsili run için tracked+full+diag; kalan run'lar sadece tablo (+ appendix).
- [ ] IMU-only drift: datasetleri tek figürde overlay (drift başlama zamanı kıyası, EuRoC dahil).
