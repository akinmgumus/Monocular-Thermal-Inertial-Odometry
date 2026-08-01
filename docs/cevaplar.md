# Background — Soru/Cevap Notları

Tez Background bölümünü okurken çıkan kavramsal soruların açıklamaları.
Bölüm bölüm ilerliyoruz; her yeni tartışma buraya eklenir.
(Formüller LaTeX; VSCode'da "Markdown+Math" eklentisi veya GitHub önizlemesiyle render olur.)

---

## sec:bg_frames — Notation and Coordinate Frames

### 1. Yer çekimi işareti ve specific force

**Kilit kavram: İvmeölçer "ivme" ölçmez, "özgül kuvvet" (specific force) ölçer** —
yani yer çekimi *dışındaki* kuvvetlerin birim kütleye düşen kısmını.

Dünya çerçevesinde gerçek (kinematik) ivme:

$$\dot{\mathbf{v}} = \mathbf{R}_I^W \mathbf{a}_{body} - \mathbf{g}^W$$

İki ayrı vektörü karıştırma:

- **Yer çekimi ivmesi** $\mathbf{g}_{grav} = [0,\,0,\,-9.81]^\top$ → aşağı (senin beklediğin, doğru).
- Koddaki $\mathbf{g}^W$ ise **bunun negatifi**: $\mathbf{g}^W = -\mathbf{g}_{grav} = [0,\,0,\,+9.81]^\top$.
  Yani "düz, dingin IMU'nun okuduğu özgül kuvvet" formunda tanımlı.

**Dingin (rest) durumda** hız sabit, $\dot{\mathbf{v}} = 0$:

$$\underbrace{\mathbf{R}_I^W \mathbf{a}_{body}}_{=\,[0,0,+g]\ (\text{IMU } +g \text{ okur})} \;-\; \underbrace{\mathbf{g}^W}_{=\,[0,0,+g]} \;=\; \mathbf{0}$$

Yani **ilk terim $[0,0,0]$ değil, $[0,0,+g]$'dir.** İkisi de $[0,0,+g]$ olduğu için
*farkları* sıfır. Denklem "her eksen sıfır" demiyor; "ölçülen özgül kuvvetten g'yi
çıkarınca gerçek ivme sıfır kalır" diyor. Senin "(0,0,g) olması gerekmiyor mu IMU'nun
gördüğü" sezgin **birebir doğru** ve sorunu çözen de bu.

$\mathbf{g}^W = [0,0,-g]$ konvansiyonu da geçerli; o zaman kinematik
$\dot{\mathbf{v}} = \mathbf{R}_I^W\mathbf{a}_{body} + \mathbf{g}^W$ olur. Biz $+g$
seçtik çünkü ivmeölçerin okuduğu değerle birebir örtüşüyor.

#### Üç senaryo: dingin / hover / serbest düşme / yukarı ivme

Senin sorduğun hover durumu burada — ama küçük bir düzeltme var: **hover sıfır okumaz, $+g$ okur.**

Genel kural: ölçülen özgül kuvvet (dünya çerçevesinde) $= \mathbf{a}_{kin} - \mathbf{g}_{grav} = \mathbf{a}_{kin} + g\hat{\mathbf{z}}$.

| Durum | Kinematik ivme $\mathbf{a}_{kin}$ | İvmeölçer okuması (özgül kuvvet) | Açıklama |
|---|---|---|---|
| **Masada duruyor** | $0$ | $+g\,\hat{\mathbf z}$ (yukarı) | Masa yukarı iter (normal kuvvet) |
| **Hover (sabit asılı)** | $0$ | $+g\,\hat{\mathbf z}$ (yukarı) | İtki (thrust) yukarı, yer çekimini dengeler — **masayla aynı** |
| **Serbest düşme** | $-g\,\hat{\mathbf z}$ | $\mathbf{0}$ | Tek kuvvet yer çekimi; destek yok → **ağırlıksız** |
| **1g ile yukarı hızlanıyor** | $+g\,\hat{\mathbf z}$ | $+2g\,\hat{\mathbf z}$ | g'yi dengelemek + 1g hızlanmak |

**Demek ki:**
- **Hover'da ivmeölçer $+g$ okur, sıfır değil** — masada durmakla *fiziksel olarak özdeş*.
  Çünkü her iki durumda da yer-çekimi-dışı kuvvet (masa tepkisi veya pervane itkisi) $+mg$ yukarı.
- **Sıfır okuduğu yer serbest düşmedir** (tek kuvvet yer çekimi, destek yok). Astronotların
  yörüngede "ağırlıksız" hissetmesi bu — sürekli serbest düşme.
- **$2g$ ise** hover değil, *1g ivmeyle tırmanış* (veya yukarı hızlanan asansör).

Senin "hover'da tersi yönde bir kuvvet mi ölçüyor" sezgin aslında doğru tarafı yakalıyor:
hover'da hissedilen kuvvet **itki** (yukarı), tıpkı masadaki normal kuvvet gibi → $+g$ yukarı.
Yani "tersi" değil, masa durumuyla **aynı** okuma. Karışıklık, hover'ı serbest düşmeyle
karıştırmaktan geliyordu.

---

### 2. Nokta dönüşümü

$$\mathbf{p}^A = \mathbf{R}_B^A\,\mathbf{p}^B + \mathbf{p}_B^A$$

Önce noktanın B'deki konumunu B→A rotasyon matrisiyle döndür, sonra B'nin orijininin
A'daki konumunu (öteleme vektörü $\mathbf{p}_B^A$) ekle. Senin anladığın gibi. ✓

---

### 3. Skew-symmetric matris

Tek işi: **cross product'ı matris çarpımına çevirmek.**

$$\lfloor \mathbf{a} \rfloor_\times \, \mathbf{b} = \mathbf{a} \times \mathbf{b},
\qquad
\lfloor \boldsymbol{\omega} \rfloor_\times =
\begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}$$

Aynı cross product (aynı $|\mathbf{a}||\mathbf{b}|\sin\theta$ büyüklüğü, aynı sağ-el yön
kuralı) — sadece lineer cebir formunda yazıyoruz ki **denklemlerin ve Jacobian'ların**
içine koyabilelim. Geometrik tanım ile cebirsel uygulama aynı şeydir.

- **Sadece 3D:** cross product bir vektör olarak yalnız 3 boyutta tanımlı → vektör 3 elemanlı olmalı.
- $\omega_1,\omega_2,\omega_3$ bizde **açısal hız** (gyro). Ama operatör geneldir.
  Kullandığımız yerler:
  - Rotasyon kinematiği: $\dot{\mathbf{R}} = \mathbf{R}\,\lfloor\boldsymbol{\omega}\rfloor_\times$
  - **Moment kolu / lever-arm:** $\mathbf{v} = \boldsymbol{\omega}\times\mathbf{r} = \lfloor\boldsymbol{\omega}\rfloor_\times\mathbf{r}$
    — senin drone örneğin **tam bu**! Kamera IMU'dan uzakta olduğu için dönüşten hız kazanır;
    augmentation Jacobian'ındaki $\lfloor\mathbf{t}_{IC}\rfloor_\times$ terimi de bu yüzden var.
  - Attitude-error Jacobian'ları (F ve H matrislerindeki $\lfloor\cdot\rfloor_\times$).

---

### 4. Quaternion

**Convention sadece sıralama mı? Hayır — iki ayrı şey var:**

- **Saklama sırası** ($[w,x,y,z]$ vs $[x,y,z,w]$): tamamen kozmetik. scipy w'yi sona koyar.
- **Çarpım convention'ı** (Hamilton vs JPL): **matematiği değiştirir** — vektör kısmının
  işareti, rotasyonların birleşme sırası, quaternion↔matris ilişkisi farklı. Karıştırırsan
  sessiz işaret hataları. Klasik MSCKF (Mourikis) **JPL**; scipy/ROS/Eigen **Hamilton**.
  Biz baştan sona **Hamilton** kullanıyoruz.

**Quaternion nedir:** birim normlu 4 sayı $\mathbf{q}=(w,x,y,z)$, aslında **eksen-açıyı**
paketler. $\mathbf{u}$ ekseni etrafında $\theta$ dönüş için:

$$w = \cos\frac{\theta}{2}, \qquad (x,y,z) = \mathbf{u}\,\sin\frac{\theta}{2}$$

x,y,z,w değerleri **böyle** hesaplanır (yarım açı).

Burada $\mathbf{u}$ = **dönüş ekseni** (birim vektör, $\lVert\mathbf{u}\rVert=1$), $\theta$ =
o eksen etrafındaki dönüş açısı. Arkasındaki fikir **Euler'in dönüş teoremi**: 3B'de
herhangi bir oryantasyon, tek bir $\mathbf{u}$ ekseni etrafında tek bir $\theta$ açılık
dönüşle ifade edilebilir. Quaternion bunu paketler: vektör kısmının ($x,y,z$) **yönü** =
eksen $\mathbf{u}$, **büyüklüğü** = $\sin(\theta/2)$, skaler kısım $w=\cos(\theta/2)$.

**Örnek** — z-ekseni etrafında $90°$: $\mathbf{u}=[0,0,1],\ \theta=90°$:

$$w = \cos 45° = 0.707, \qquad (x,y,z) = [0,0,1]\cdot\sin 45° = [0,0,0.707]$$

yani $\mathbf{q} = (0.707,\ 0,\ 0,\ 0.707)$.

**Euler'den farkı / gimbal lock:** Euler = ardışık 3 açı. Belirli oryantasyonda (ör.
pitch $=90°$) iki dönüş ekseni çakışır, bir serbestlik derecesi kaybolur (1. ve 3.
dönüşü ayıramazsın) — **tekillik**. Quaternion'da tekillik yok; her oryantasyon $S^3$
küresinde pürüzsüz bir nokta + pürüzsüz interpolasyon (slerp) + ucuz birleştirme.
"Orientation zaten 3 açı değil mi" — evet, *Euler temsili* öyle; biz tam onun
gimbal-lock derdinden kaçmak için iç temsilde quaternion kullanıyoruz.

#### Somut örnek: gimbal lock (Euler kilitlenir, quaternion kilitlenmez)

ZYX (yaw–pitch–roll) konvansiyonunda **pitch $=90°$** olunca kilitlenme olur. Burnu
dik yukarı bakan bir uçak düşün: **yaw** dünya-$z$ ekseni etrafında döndürür; **roll**
ise gövde-$x$ (burun) ekseni etrafında — ama pitch $=90°$ olunca burun da dünya-$z$'ye
bakıyor. Yani **yaw ve roll aynı eksende** → sadece farkları ($\mathrm{yaw}-\mathrm{roll}$)
gözlemlenebilir, bir serbestlik derecesi kaybolur (3 DOF → 2).

İki **farklı** Euler üçlüsü, **aynı** oryantasyonu verir (her ikisinde de $\mathrm{yaw}-\mathrm{roll}=0$):

- A: $(\mathrm{yaw},\mathrm{pitch},\mathrm{roll}) = (0°,\ 90°,\ 0°)$
- B: $(\mathrm{yaw},\mathrm{pitch},\mathrm{roll}) = (40°,\ 90°,\ 40°)$

İkisinin de rotasyon matrisi birebir aynı:

$$R_A = R_B = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix}$$

Bu matristen **geri** Euler'e çevirince ikisi de $(0°, 90°, 0°)$ döner — girilen
$40°/40°$ **geri çıkarılamıyor**. İşte senin bahsettiğin "backward kinematik tıkanıyor"
durumu: sonsuz sayıda $(\mathrm{yaw},\mathrm{roll})$ çifti aynı oryantasyonu verir,
ayrıştırılamaz.

**Quaternion'da sorun yok:** her iki durum da aynı **tek** quaternion'u verir,

$$\mathbf{q} = (w,x,y,z) = (0.707,\ 0,\ 0.707,\ 0),$$

ki bu sadece "$y$ ekseni etrafında $90°$" demek ($\mathbf{u}=[0,1,0],\ \theta=90°$) —
tekil değil, pürüzsüz, tek anlamlı. (scipy ile sayısal olarak doğrulandı.)

---

### 5. Exponential map  $\mathrm{Exp}(\boldsymbol{\phi}) = \exp(\lfloor\boldsymbol{\phi}\rfloor_\times)$

**Problem:** Bir dönüşü iki şekilde yazabiliriz:

1. **Rotasyon vektörü** $\boldsymbol{\phi}$ — tek bir 3-vektör. Yönü = dönüş ekseni,
   büyüklüğü = dönüş açısı ($\boldsymbol{\phi} = \theta\,\mathbf{u}$). Kompakt, 3 sayı.
   (Not: bu *tek eksende tek dönüş*; Euler'in ardışık 3 açısı değil.)
2. **Rotasyon matrisi** $\mathbf{R}$ — vektörleri döndürmek için çarptığımız 3×3 matris.

**Exponential map = bu ikisi arasındaki köprü:** bir rotasyon vektörünü alır, karşılık
gelen rotasyon matrisini üretir. $\mathrm{Exp}: \mathbb{R}^3 \to SO(3)$.

**Neden lazım:** Filtrede dönüşler doğal olarak küçük 3-vektör halinde çıkar (gyro
$\boldsymbol{\omega}\Delta t$ verir; error-state $\delta\boldsymbol{\theta}$ bir 3-vektör).
Ama bir şeyi gerçekten döndürmek / iki dönüşü birleştirmek için matris (veya quaternion)
gerekir. Exp, vektörü matrise çevirir.

**Notasyondaki $\exp$ skaler $e^x$ değil, matris üsteli:** bir matrisin seri açılımı,
$\exp(M) = \mathbf{I} + M + \tfrac{1}{2!}M^2 + \tfrac{1}{3!}M^3 + \cdots$. İçine skew
matrisini koyarız çünkü dönüşlerin "üreteci" antisimetrik matristir; bir skew matrisinin
üsteli **daima bir rotasyon matrisi** verir. Sonsuz seriyi hesaplamayız — **kapalı formu
(Rodrigues)** var:

$$\mathrm{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \sin\theta\,\lfloor\mathbf{u}\rfloor_\times + (1-\cos\theta)\,\lfloor\mathbf{u}\rfloor_\times^2,
\qquad \theta = \lVert\boldsymbol{\phi}\rVert,\ \ \mathbf{u} = \boldsymbol{\phi}/\theta$$

**Sezgi:** Skaler $\dot{x} = a\,x$ denkleminin çözümü $x(t) = x(0)\,e^{at}$ idi. Dönüş
kinematiği $\dot{\mathbf{R}} = \mathbf{R}\,\lfloor\boldsymbol{\omega}\rfloor_\times$ de
aynı yapıda; çözümü $e^{at}$'nin **matris karşılığı** olan matris üsteli. Yani Exp,
"rotasyonun $e^{at}$'si".

**Sayısal örnek — $z$ ekseni etrafında $90°$:**

$\boldsymbol{\phi} = \theta\,\mathbf{u} = \tfrac{\pi}{2}[0,0,1] = [0,\,0,\,1.5708]$,
yani $\theta = 90°$, $\mathbf{u} = [0,0,1]$. Önce skew ve karesi:

$$\lfloor\mathbf{u}\rfloor_\times = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix},
\qquad
\lfloor\mathbf{u}\rfloor_\times^2 = \begin{bmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

$\sin 90° = 1$ ve $1 - \cos 90° = 1$ olduğundan:

$$\mathrm{Exp}(\boldsymbol{\phi}) = \mathbf{I} + 1\cdot\lfloor\mathbf{u}\rfloor_\times + 1\cdot\lfloor\mathbf{u}\rfloor_\times^2
= \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

ki bu tam olarak $z$ etrafında $90°$ döndüren standart matris $R_z(90°)$. ✓
($[0,0,1.5708]$ rotasyon vektörü → $R_z(90°)$ matrisi.)

**$\mathrm{Log}$** tam tersidir: rotasyon matrisini alır, geri rotasyon vektörünü
($\boldsymbol{\phi}$, eksen-açı) verir.

#### "Ama bunu quaternionla da yapabiliyoruz, Exp niye?"

Exp ile quaternion **rakip değil**: quaternion oryantasyonu *nasıl sakladığımız* (4 sayı,
gimbal-lock yok); Exp/Log ise filtrenin **3-vektör dünyası** ile **manifold $SO(3)$**
arasındaki *geçiş işlemi*.

"Örneği quaternionla yaptık" demek aslında **Exp'i kullanmak** demektir: scipy'de
`R.from_rotvec(φ)` doğrudan $\mathrm{Exp}(\boldsymbol\phi)$'dir (Rodrigues'i hesaplar,
sonucu quaternion olarak verir). **Exp = işlem, quaternion = kap.**

**Neden gerçekten gerekli:** Kalman filtresi hatayı/kovaryansı **minimal 3-vektör**
$\delta\boldsymbol\theta$ olarak tutmak zorunda (4-sayı quaternion kovaryansı tekil olur).
Ama gerçek oryantasyon $SO(3)$'te yaşar. Bu ikisini bağlamak için köprü şart:

- **Exp** ($\mathbb{R}^3 \to SO(3)$): gyro artışını entegre etmek ve Kalman'ın
  $\delta\boldsymbol\theta$ düzeltmesini oryantasyona **enjekte** etmek için.
- **Log** ($SO(3) \to \mathbb{R}^3$): iki oryantasyon arasındaki **hatayı**
  $\delta\boldsymbol\theta$ olarak çıkarmak için.

Quaternion tek başına "3-sayılık düzeltmeyi oryantasyona nasıl uygularım"ı çözmez — onu
Exp çözer. Kodda her güncellemede kullanılıyor:

- `dq = R.from_rotvec(omega*dt)` → $\mathrm{Exp}(\boldsymbol\omega\Delta t)$, gyro propagation (`msckf.py:255`)
- `nominal_rot = nominal_rot * R.from_rotvec(delta_x[0:3])` → $\delta\boldsymbol\theta$ düzeltme enjeksiyonu (`msckf.py:755`)

#### Ek netleştirmeler (Exp)

**"x'te 30° → matris" sonucunu Exp mi veriyor?** Evet. $\boldsymbol\phi=[0.5236,0,0]$
(30° × x) → $\mathrm{Exp}(\boldsymbol\phi)=R_x(30°)=\begin{bmatrix}1&0&0\\0&0.866&-0.5\\0&0.5&0.866\end{bmatrix}$ (scipy ile doğrulandı).

**IMU→world geçişinde mi kullanıyoruz?** İnce ayrım:
- *Geçişin kendisi* sadece matrisle çarpmak: $\mathbf{v}^W=\mathbf{R}_I^W\mathbf{v}^I$ (`accel_world = R @ accel`). Exp yok.
- *Exp'i* o $\mathbf{R}_I^W$ matrisini **inşa/güncelleme** için kullanıyoruz (gyro entegrasyonu + δθ düzeltmesi). Yani Exp matrisi üretir → o matris geçişte kullanılır.

**"3-vektör dünyası" ne?** Filtrede iki tür nicelik var: (a) **düz $\mathbb{R}^3$** üçlüleri
(gyro $\boldsymbol\omega$, hata $\delta\boldsymbol\theta$, düzeltme) — kısıtsız, serbestçe toplanır;
(b) **oryantasyon** — eğri yüzeyde ($SO(3)$), öyle 3 sayı eklenemez. "3-vektör dünyası" = (a).
Exp düz uzaydan eğri uzaya köprü, Log ters yön.

**Tek eksende Euler de yapar, neden Exp?** Tek eksen için Euler yeter, ama girdilerimiz
tek eksen değil: gyro genelde 3 bileşenli (keyfi eksen), δθ keyfi 3-vektör. Exp keyfi-eksen
dönüşünü tek hamlede, tekillik-siz, birleşebilir biçimde verir; Euler gimbal-lock'a düşer.
Özel-durum açmıyoruz — her keyfi 3-vektör için aynı Exp.

---

### 6. Sağdan çarpımla yayma  $\mathbf{R}_{I,k+1}^W = \mathbf{R}_{I,k}^W\,\mathrm{Exp}(\boldsymbol{\omega}\Delta t)$

- Gyro **body-frame** açısal hız $\boldsymbol{\omega}$ ölçer. $\Delta t$ içinde cisim,
  ekseni $\boldsymbol{\omega}$ yönünde, açısı $|\boldsymbol{\omega}|\Delta t$ olan küçük
  bir dönüş yapar → rotasyon vektörü $\boldsymbol{\omega}\Delta t$.
- $\mathrm{Exp}(\boldsymbol{\omega}\Delta t)$ bu küçük dönüşü matrise çevirir.
- $\boldsymbol{\omega}$ **body** frame'de olduğu için artış **sağdan** çarpılır.
  (Kural: body artışı → sağdan; world artışı → soldan.)
- Bu aslında $\dot{\mathbf{R}} = \mathbf{R}\,\lfloor\boldsymbol{\omega}\rfloor_\times$
  kinematiğinin (sabit $\boldsymbol{\omega}$ için) **tam** ayrık çözümü — $\exp$ tam o
  yüzden doğru değeri verir.

---

### 7. Nominal state + error state

**Error-state (indirect) Kalman filtresi.** State ikiye ayrılır:

- **Nominal state** $\hat{\mathbf{x}}$ = şu anki **en iyi tahmin** (büyük değerler,
  doğrusal-olmayan hareketle entegre edilir). Bu $\dot{x}$ **değil** — $\hat{x}$, x'in
  tahmini (türev değil).
- **Error state** $\tilde{\mathbf{x}}$ = gerçek ile nominal arasındaki **küçük fark**.
  Kalman aslında *bu hatayı* (ortalama ~0, kovaryans) izler, sonra nominal'e enjekte
  edip sıfırlar.

$$\mathbf{x} = \hat{\mathbf{x}} + \tilde{\mathbf{x}}
\qquad(\text{gerçek} = \text{tahmin} + \text{hata})$$

$\mathbf{x}$ gerçek (bilinmeyen), $\hat{\mathbf{x}}$ tahminimiz, $\tilde{\mathbf{x}}$
filtrenin istatistiksel izlediği küçük hata.

**Neden:** hata küçük → EKF lineerizasyonu geçerli; kovaryans minimal (15-boyut);
rotasyonun manifold yapısı temiz ele alınır.

**15 state:** 3 oryantasyon + 3 hız + 3 konum + 3 gyro bias + 3 accel bias = 15. ✓
Bunların **12'si toplamsal** hata (pos, vel, iki bias), **3'ü (oryantasyon) çarpımsal**.

**Neden oryantasyon farklı:** rotasyon düz vektör uzayında değil, **eğri manifold'da**
($SO(3)$) yaşar. Rotasyona 3-vektör *eklersen* geçerli rotasyon kalmaz. O yüzden
**çarpımsal hata**:

$$\mathbf{R}_I^W = \hat{\mathbf{R}}_I^W\,\mathrm{Exp}(\delta\boldsymbol{\theta})$$

Küçük dönüş 3 sayılık $\delta\boldsymbol{\theta}$ ile parametrelenir, hep geçerli kalır.

**Quaternion neden sadece oryantasyon için + neden 4 yerine 3:**

- **Nominal** oryantasyon quaternion (4 sayı) → entegrasyonda gimbal-lock yok.
- **Hata/kovaryans** için 4 sayı kullanırsak: quaternion'un 4 sayısı ama **3 serbestlik
  derecesi** var (birim-norm 1'ini siler). 4×4 kovaryans **rank-eksik (tekil)** olur →
  sayısal bela. O yüzden hatayı **minimal 3-parametreli $\delta\boldsymbol{\theta}$** ile
  tutuyoruz → 3×3, tam-rank.
- Yani "4'ten 1'inden kurtulduk" = redundant 4-sayı yerine teğet-uzayda minimal 3-DOF.
  Pos/vel/bias sıradan vektör → onlarda quaternion'a gerek yok.

---

## sec:bg_imu — Inertial Measurement Model

Bu bölüm, kurduğumuz notasyonu (Exp, 15-DOF error-state) alıp **IMU'nun hareketi nasıl
ilerlettiğini** matematikleştirir. Sonunda doğrudan koddaki `predict()` ve `F` matrisi çıkar.
**Her formülün hemen altında sembollerini açıklıyorum.**

### 1. IMU ölçüm modeli — gyro ve ivmeölçer gerçekte ne ölçer

$$\boldsymbol{\omega}_m = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g, \qquad
\mathbf{a}_m = \mathbf{a} + \mathbf{b}_a + \mathbf{n}_a$$

Semboller:
- $\boldsymbol{\omega}_m,\ \mathbf{a}_m$ — gyro ve ivmeölçerin **ham** ölçümü (body frame).
- $\boldsymbol{\omega}$ — gerçek açısal hız; $\mathbf{a}$ — gerçek **özgül kuvvet** (body).
  Sensör doğrudan bunu ölçer → ölçüm modeli sadece "gerçek + bias + gürültü".
- $\mathbf{b}_g,\ \mathbf{b}_a$ — gyro / accel **bias** (yavaşça kayan ofset, body).
- $\mathbf{n}_g,\ \mathbf{n}_a$ — gyro / accel **beyaz gürültü**.
- $\mathbf{g} = [0,0,{+}9.81]$ — yer çekimi. **Ölçüm modelinde değil**, kinematikte
  devreye girer ($\dot{\mathbf{v}} = \mathbf{R}\mathbf{a} - \mathbf{g}$, part-2).
  Dingin'de $\mathbf{a} = \mathbf{R}_I^{W\top}\mathbf{g}$ (Q1).

Filtrede tahmini bias'ı çıkarıp kullanırız: $\boldsymbol{\omega} = \boldsymbol{\omega}_m - \hat{\mathbf{b}}_g$,
$\mathbf{a} = \mathbf{a}_m - \hat{\mathbf{b}}_a$ (kod 251–252).

### 2. Sürekli-zaman kinematiği — state nasıl değişir

$$\dot{\mathbf{R}} = \mathbf{R}\lfloor\boldsymbol{\omega}\rfloor_\times, \quad
\dot{\mathbf{v}} = \mathbf{R}\mathbf{a} - \mathbf{g}, \quad
\dot{\mathbf{p}} = \mathbf{v}, \quad
\dot{\mathbf{b}}_g = \mathbf{n}_{bg}, \ \dot{\mathbf{b}}_a = \mathbf{n}_{ba}$$

Semboller:
- $\mathbf{R} = \mathbf{R}_I^W$ — oryantasyon (IMU→world); $\mathbf{v}$ — hız, $\mathbf{p}$ — konum (world).
- $\dot{(\cdot)}$ — zaman türevi ("nasıl değişiyor").
- $\mathbf{n}_{bg},\ \mathbf{n}_{ba}$ — bias **random walk** (bias'ı zamanla kaydıran gürültü).

Satır satır: oryantasyon açısal hızla döner; hız dünya-ivmesi $\mathbf{R}\mathbf{a}-\mathbf{g}$
ile değişir; konum hızla; bias'lar random walk ile yavaşça kayar.

**$\dot{\mathbf{R}} = \mathbf{R}\lfloor\boldsymbol{\omega}\rfloor_\times$ tam olarak ne?**
Rotasyon kinematiği — oryantasyonun açısal hızla nasıl değiştiğini veren denklem. Türetme:
gövdeye sabit bir noktanın world konumu $\mathbf{p}_W = \mathbf{R}\,\mathbf{p}_{body}$
($\mathbf{p}_{body}$ sabit), hızı $\dot{\mathbf{p}}_W = \dot{\mathbf{R}}\,\mathbf{p}_{body}$.
Ama fizikçe dönen nokta $\boldsymbol{\omega}\times\mathbf{p}$ hızıyla hareket eder:
$\dot{\mathbf{p}}_W = \mathbf{R}(\boldsymbol{\omega}\times\mathbf{p}_{body}) = \mathbf{R}\lfloor\boldsymbol{\omega}\rfloor_\times\mathbf{p}_{body}$.
İkisini eşitle → $\dot{\mathbf{R}} = \mathbf{R}\lfloor\boldsymbol{\omega}\rfloor_\times$. (Skew
kullanırız çünkü rotasyonun değişimi vektör toplamı değil, antisimetrik "büküm".) Bunun
sabit-$\boldsymbol{\omega}$ tam çözümü zaten part-3'teki $\mathbf{R}_{k+1} = \mathbf{R}_k\mathrm{Exp}(\boldsymbol{\omega}\Delta t)$.

### 3. Ayrık-zaman propagation — kamera kareleri arası (= `predict()`)

$$\mathbf{R}_{k+1} = \mathbf{R}_k\,\mathrm{Exp}(\boldsymbol{\omega}\Delta t)$$
$$\mathbf{v}_{k+1} = \mathbf{v}_k + (\mathbf{R}_k\mathbf{a} - \mathbf{g})\Delta t$$
$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k\Delta t + \tfrac{1}{2}(\mathbf{R}_k\mathbf{a} - \mathbf{g})\Delta t^2$$

Semboller:
- alt-indis $k$ — adım numarası ($k{+}1$ = bir sonraki örnek).
- $\Delta t$ — iki örnek arası süre.
- $\mathrm{Exp}$ — exponential map (gyro artışı $\boldsymbol{\omega}\Delta t$'yi rotasyona çevirir).

Bunlar part-2'deki sürekli denklemlerin (birinci-derece Euler) **integrali** — koddaki
`predict()` (satır 255, 266–269) birebir budur.

### 4. Error-state dinamiği → `F` matrisi — hata nasıl yayılır

$$\dot{\delta\boldsymbol{\theta}} = -\lfloor\boldsymbol{\omega}\rfloor_\times\,\delta\boldsymbol{\theta} - \delta\mathbf{b}_g$$
$$\dot{\delta\mathbf{v}} = -\mathbf{R}\lfloor\mathbf{a}\rfloor_\times\,\delta\boldsymbol{\theta} - \mathbf{R}\,\delta\mathbf{b}_a$$
$$\dot{\delta\mathbf{p}} = \delta\mathbf{v}$$

Semboller:
- $\delta\boldsymbol{\theta},\ \delta\mathbf{v},\ \delta\mathbf{p},\ \delta\mathbf{b}_g,\ \delta\mathbf{b}_a$
  — hata bileşenleri (3'er, toplam **15-DOF error-state**).
- $\mathbf{F}$ — bu denklemleri toplayan **15×15 error-state geçiş matrisi**.

Bunlar küçük-açı lineerizasyonundan gelir ve doğrudan `F`'in bloklarıdır (kod 256–264).
$\mathbf{R}$ burada FEJ ile dondurulur (detay Method'da).

### 5. Kovaryans yayılımı — belirsizlik nasıl büyür

$$\Phi = \mathbf{I} + \mathbf{F}\Delta t, \qquad \mathbf{P}_{k+1} = \Phi\,\mathbf{P}\,\Phi^\top + \mathbf{Q}$$

Semboller:
- $\Phi$ — ayrık geçiş matrisi ($\mathbf{F}$'ten); $\mathbf{P}$ — state **kovaryansı** (belirsizlik).
- $\mathbf{Q}$ — **süreç gürültüsü** kovaryansı; IMU gürültü σ'larından kurulur:
  - $\sigma_g,\ \sigma_a$ — gyro / accel **noise density** (beyaz gürültü şiddeti).
  - $\sigma_{bg},\ \sigma_{ba}$ — gyro / accel **bias random-walk** oranı.

Bu dört σ datasete göre değişir (Allan variance ile karakterize vs sadece datasheet) —
**RQ3(b)** bunun drift'e etkisini ölçer.

### Background ↔ Method sınırı

Burada *genel* model var (her IMU-tabanlı ESKF'in ihtiyacı). *Bizim* spesifik seçimler —
birinci-derece Euler discretization, gürültü değerlerinin sayısal provenance'ı, FEJ
anchoring implementasyonu — Method'a kalır.

### Neye bağlanıyor

Bu bölüm `predict()` ve `F` matrisini üretir. Sonra MSCKF bölümü (kamera state ekleme +
null-space update) bunun üstüne gelir.
