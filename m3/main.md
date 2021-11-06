% Çokkiplilik ve Çokgörevlilik
% M. Yusuf Sarıgöz
% 11/06/2021


## Tekgörevliliğin problemleri
- Aşırı uyumlama
- Yetersiz veri
- Aşırı dar problem tanımı
- Değişen koşullara yüksek hassasiyet
- ve daha fazlası

## Çokgörevlilikte ilk adım
- Multimodel by Google Brain (2017) ile çokgörevlilikte ilk adımlar atıldı.
- Her biri 8 ayrı görevden birini yerine getirmek için birlikte eğitilen modellerden oluşuyordu.
- Farklı kiplerdeki verilerle aynı anda eğitim yapılması alt modellerin performansını arttırdı.
- Verinin yetersiz olduğu görevlerde performans artışı daha yüksekti.
- Yöntemin kolayca uyarlanabilir ve genişletilebilir olmamasından ötürü fazla popülerleşmedi.
- https://ai.googleblog.com/2017/06/multimodel-multi-task-machine-learning.html

## Çokgörevliliğin ilk başarıları
- BERT by Google Brain (2018) ve benzeri modeller ile çokgörevlilik daha fazla konuşulur oldu.
- Bu modeller kolay genişletilebilirlik problemini çözdüyse de kolay uyarlanabilirlik problemi aynı kaldı.
 Her görev için yeni bir mimarinin ana gövdeye eklenmesi gerekiyordu.

## Çokgörevlilikte yeni adımlar
- T5 by Google Brain (2020) ile tüm görevleri "metinden metine" bir görev olarak tasarlamak mümkün oldu.
- İşte gerçek çokgörevlilik bu! 😄🥳
- Önek mekanizması, çokgörevliliği "utanç verici şekilde basit" hale getirdi. 🤠🇺🇸
- https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html

## Çokgörevlilik hayaldi gerçek oldu! 😜
- Multimodal Unified Model (MUM) by Google (2021) T5 modelini 75 dile genişletti.
- Metnin yanında görselleri de işleyebiliyor.
- Bir dilde ya da kipte olmayan bir bilgiyi başka bir dil ya da kipten getirebiliyor.
- Üretici bir model olmasından ötürü performansı değerlendirilmeye devam ediliyor.
- https://blog.google/products/search/introducing-mum/

## Biraz da ayaklarımız yere bassın 🙏
- CLIP by Open AI ile metin ve görsellerin aynı vektör uzayında temsillerini elde etmek mümkün oldu.
- Çokkipli ve çokgörevli bu model ile zero-shot işlemlerin uygulama alanları genişledi.
- "Give me a latent vector space, and I can move the world." 🤓

## Ek okumalar
- Multitasking and Multimodal Models in Turkish: https://huggingface.co/spaces/mys/m3tr

## E-mail
yusufsarigoz@gmail.com
