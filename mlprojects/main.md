% Executing ML Projects
% M. Yusuf Sarıgöz
% 05/01/2021

## ML'den önce
- ML'siz bir ürün çıkarmaktan korkma.
- Veri toplama konusunda yasaların imkan verdiği fırsatları değerlendir.
- Uzun ML süreçlerine başlamadan önce basit bir höristik kullan.
- Karmaşık bir höristik yerine ML'i bekle.

## ML projeleri neden farklı?
- Doğası gereği deneyseldir.
- Döngüseldir.
- Farklı kaynak, araç ve yeteneklerin bir arada bulunmasını ve bunların yönetimini gerektirir.

## Problemi tanımla
- Seni ne rahatsız ediyor? (pain point)
- Problemin çözümü için hangi kaynakları kullanabilirsin?
- Problemin çözümü nasıl mümkün olabilir? Sınıflandırma, regresyon, üretim vb.
- Başarı ölçütün nedir?

## İhtiyacı sorgula!
- Gerçekten makine öğrenmesine ihtiyacın var mı?
- Klasik algoritmaları neden kullanamıyorsun?
- Baseline yöntemler ve başarı oranları?

## Hipotezini kur
::: sorular
- Problemi çözmek için ne yapmalısın?
- Nasıl bir veriye ihtiyacın var?
- Bu veriyi nasıl elde edebilirsin?
- Nasıl bir modelleme problemini çözer?
:::
::: great
- Projeyi hâlâ yapılabilir görüyorsan harika!
:::

#### Veri topla
- *Ne kadar?* Bilmiyoruz. Şu an için ne kadar fazla, o kadar iyi.
- *Nereden?* Açık veriler varsa harika. Yoksa kendimiz kazıyabilir miyiz?
- *Bizim elimizde bir veri var mı?*
- Dışarıdan gelen veriyle elimizdeki *veriyi nasıl birleştirebiliriz?*

## Veriyi hazırla
- Görselleştir. Veriyi tanımalı.
- Kullanılabilir olması için nasıl dönüştürülmeli?
- Etiketlenmesi gerekiyor mu?
- Verinin tamamına ihtiyacın var mı?
- Veriyi önişlemeden geçir.
- Tüm adımlar belgelendirilmeli.

## Veriyi altkümelere ayır
::: subsets
- Eğitim kümesi
- Tes kümesi
- Geçerleme kümesi
:::
::: note
- Hepsi aynı kaynaktan
- ve dağılımları benzer olmalı.
:::

## Modelleme
- Birden fazla model eğitilecek.
- Küçük ve basit bir modelle başla.
- Zamanla karmaşıklığı arttır.
- Performans artışı, modele eklenen karmaşıklığa değiyor mu?

## Modelleri değerlendir
- Başarı metrikleri, model büyüklüğü, toplam tepki süresi düşünüldüğünde en iyisi hangisi?
- Cross-validation ile hyperparameter tuning yapabilir miyiz?
- Baseline yöntemlerle karşılaştır.

## İlk pipeline yapını devreye al
- Bu, ML projesindeki en büyük adım olacak.
- Veride yapılacak öznitelik değişikliklerini hızlıca deneyebileceğimiz CI/CD/CT yapısı.
- CI/CD altyapıyı ML'den bağımsız şekilde üretime almalı.
- ML çalışmaları CT sürecini tetiklemeli.

## Model optimizasyonu
- Model merkezli değil veri merkezli düşün.
- Veride yapılacak öznitelik mühendislikliği daha doğrudan bir performans artışı sağlayabilir.
- Yeni öznitelikler yarat ve denemekten korkma.
- ML'den önceki höristikleri özniteliklere dönüştür.
- Gerektiğinde veri toplama adımına geri dönmekten çekinme.
- Model performansını takip et ve gerektiğinde modeli yeni verilerle tekrar eğit.

## Ek okumalar
Rules of Machine Learning https://bit.ly/3vtQpVa
- Organizing Machine Learning Projects: Project Management Guidelines https://bit.ly/3xDW8K7
- Machine Learning: The high-interest credit card of technical debt https://bit.ly/3t9pVXF

## E-mail
yusuf@ailabs.com.tr
