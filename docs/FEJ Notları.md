# FEJ Notları

MSCKF pipeline'ında **First-Estimate Jacobians (FEJ)** için yaptıklarımız,
denediklerimiz ve aldığımız sonuçların basit-dilli kaydı. Yeni FEJ bulguları
**hep buraya** eklenir.

---

## 1. FEJ nedir, neden gerekti

- VIO'da state'in bazı yönleri **gözlemlenemez**: global pozisyon (3) ve global
  yaw (yerçekimi etrafındaki dönüş, 1) — toplam **4 yön**.
- Standart EKF, Jacobian'ı **her adımda güncel tahminde** lineerize eder.
  Lineerizasyon noktası sürekli değiştiği için filtre bu gözlemlenemez yönlerde
  **sahte bilgi kazanır** → kovaryans gereksiz küçülür → **aşırı güven**, yaw
  drift'i, tutarsızlık (NIS bozulur).
- **FEJ fikri:** Jacobian'ı, ilgili state'in **filtreye ilk girdiği andaki**
  tahmininde dondur ve bir daha değiştirme. Residual ise **güncel** tahminde
  kalsın. Böylece gözlemlenemez alt-uzay korunur, sahte düzeltme döngüsü kapanır.

---

## 2. Standart Jacobian vs bizim FEJ — fark hangi değişkenlerde

Tek fark: Jacobian/kovaryans yolunda **`_fej` çapalı** değişkenleri kullanmak.
Gerçek state entegrasyonu ve residual **güncel** kalır.

| Nerede | Standart EKF | Bizim FEJ |
|---|---|---|
| Propagation `F` (hız↔attitude, hız↔bias blokları) | `nominal_rot` | **`nominal_rot_fej`** |
| Measurement Jacobian `H_x`, `H_f` | `cs.rot`, `cs.pos` | **`cs.rot_fej`, `cs.pos_fej`** |
| Jacobian'daki landmark | `p_f_world` | **`p_f_world_fej`** (çapalanmış) |
| Residual `z − ẑ` | güncel | güncel (aynı) |
| Nominal vel/pos entegrasyonu | güncel | güncel (aynı) |

Özet: **FEJ yalnız Jacobian'ı dokunur**, gerçek tahmini değil.

---

## 3. Bizim yaptıklarımız

1. **Cam-state FEJ:** her kamera pozu state'e eklenince (`augment`) o anki pozu
   `cs.rot_fej`, `cs.pos_fej` olarak **donduruyoruz**; asla değişmez.
   (`msckf.py:66-71`)
2. **IMU-state FEJ:** `nominal_rot_fej` çapası, propagation `F` matrisinin
   rotasyon-bağı bloklarında kullanılıyor (`msckf.py:260-261`). Static-init
   sonrası bir kez tazeleniyor (`msckf.py:211-213`).
3. **Landmark anchoring (demirleme):** GN, landmark'ı kameraların **güncel**
   pozlarıyla üçgenliyor. Bunu doğrudan FEJ pozlara iz düşürmek geometrik
   tutarsızlık yaratıyordu. Çözüm: landmark'ı çapa kameranın (ilk gözlem)
   **güncel→FEJ rijit dönüşümüyle** "FEJ dünyasına" demirleyip Jacobian'da onu
   kullanmak (`p_f_world_fej`, `msckf.py:492-495`).
4. **Residual güncel, Jacobian FEJ:** her view için Jacobian FEJ pozda, residual
   (`z − ẑ`) güncel pozda hesaplanıyor (`msckf.py:531-545`).

---

## 4. Denediklerimiz (ve neden vazgeçtik)

- **Option 3 — cam-FEJ'i IMU-FEJ çapasına zincirleme:** kamera first-estimate'ini
  IMU FEJ çapasından türetmeyi denedik; **ampirik olarak ters tepti**, geri
  alındı. Cam state kendi **bağımsız** first-estimate'iyle kalmalı.
  (`msckf.py:67-70`)
- **`nominal_pos_fej` + cam-FEJ chaining:** pozisyonu da FEJ'e bağlamayı denedik;
  geri alındı (`msckf.py:131-132`).
- **Sonuç olarak benimsenen:** IMU tarafında **yalnız rotasyon** çapası
  (`nominal_rot_fej`), cam tarafında **bağımsız** poz çapası (`rot_fej/pos_fej`).

---

## 5. Sonuçlar

- **İyi yön:** filtre tutarlılığı düzeldi — yaw drift'i azaldı, sahte attitude
  düzeltme döngüsü engellendi, NIS / `used/in` oranı daha sağlıklı.
- **Sınır:** FEJ bir **lineerizasyon-tutarlılığı** düzeltmesidir; **gerçek**
  gözlemlenebilirlik kaybını çözmez. Örn. ROVTIO ~118 s'deki katastrofik
  divergence FEJ ile düzelmiyor — orada görsel kısıtlar fiilen kayboluyor (ayrı
  problem, IMU dead-reckoning).

---

## 6. Kodda nasıl bulunur (organizasyon)

FEJ doğası gereği **tek fonksiyona toplanamaz** — üç ayrı Jacobian'a (propagation
`F`, augmentation `J_imu`, measurement `H`) dokunan bir ilke. Taşımak yerine
**okunaklı hale getirdik** (davranış değişmedi):

- `MSCKF` sınıf docstring'inde **"FEJ design map"** bloğu — tüm dokunuş noktalarını
  tek yerde özetler (tek okunaklı referans).
- Her FEJ satırı **`# [FEJ]`** ile etiketli → `grep -nE "\[FEJ\]|_fej" msckf.py`
  hepsini tek listede verir.

## 7. Kod referansları

- Propagation FEJ (F matrisi): `src/thermal_vo/thermal_vo/msckf.py:238-261`
- CamState FEJ snapshot: `msckf.py:66-71`
- IMU-state FEJ tazeleme: `msckf.py:211-213`
- Update FEJ anchoring + Jacobian: `msckf.py:444-552`