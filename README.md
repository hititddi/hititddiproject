<p align="center"><a href="https://www.teknofest.org" target="_blank"><img src="https://github.com/hititddi/hititddiproject/blob/main/logo.svg" width="400"></a></p>

# Giriş:

Bu proje, GNU Affero Genel Kamu Lisansı v3 (AGPLv3) altında lisanslanmıştır.

Bu lisans, herhangi bir kişinin, değiştirebileceği, dağıtabileceği veya ticari amaçlar için kullanabileceği açık kaynak kodlu yazılımların geliştirilmesini ve yayılmasını teşvik etmek için tasarlanmıştır. Aynı zamanda, tüm değişikliklerin kaynak kodunun dağıtımı gerektiği için, topluluğun yararına olan geliştirmelerin tekrar kullanılabilir olmasını sağlar.

Bu README dosyası, proje hakkında ayrıntılı bilgi sağlar ve projenin kurulumu, kullanımı ve katkıda bulunulması hakkında talimatlar içerir.

# Projenin Amacı:

Bu proje Türkçe doğal dil işlemesi için kullanıcı dostu ve yüksek performanslı kütüphanelerin ve veri kümelerinin hazırlanmasına katkı sağlamayı amaçlamaktadır. Bu amaç doğrultusunda "Aşağılayıcı Söylemlerin Doğal Dil İşleme İle Tespiti" problemiyle bir yarışma düzenlenmektedir. Yarışmanın amacı, platform ekosisteminin gelişimine katkıda bulunacak bir veri seti ve problem belirleyerek, Türkçe doğal dil işleme alanındaki gelişmeleri teşvik etmektir.

HititDDİ ekibi olarak projenin amacı ise Türkçe aşağılayıcı söylemlerin doğal dil işleme ile yüksek doğruluklu tespitini sağlamaktadır.

# Kurulum:

[Projenin nasıl kurulacağına dair adım adım talimatları verin. Gereksinimler ve bağımlılıklar hakkında bilgi verin.]

1. Öncelikle, diğer projelerle karışmaması adına yeni bir conda çevresi oluşturulması tavsiye edilir.

    ```bash
    conda create -n nyp python=3.9
    conda activate nyp
    ```

2. Ardından aşağıda belirtilen projelerin bağımlılıklarını yüklemeniz gerekir.

    ```bash
    pip3 install numpy
    pip3 install transformers
    pip3 install scikit-learn
    pip3 install seaborn
    pip3 install matplotlib
    ```

3. En sonunda ise torch bağımlılığın kurulması gerekir. Bağımlılığın diğer sürümleri için lütfen [tıklayınız](https://pytorch.org/).

    ```bash
    pip3 install torch --index-url https://download.pytorch.org/whl/cu118
    ```

# Kullanım:

1. Verisetini eğitmek için aşağıdaki scripti giriniz.

    ```bash
    python train.py
    ```

2. Metin verilerini tespit etmek için aşağıdaki scripti giriniz.

    ```bash
    python predict.py testverisi.csv cikti.csv binary_model.pt multi_model.pt
    ```

# Katılım:

Bu projeye katkıda bulunmak isterseniz, lütfen aşağıdaki adımları takip edin:

1. Projeyi kopyalayın veya klonlayın.

    ```bash
    git clone https://github.com/hititddi/hititddiproject.git
    ```

2. Değişikliklerinizi yapın ve değişikliklerinizi göndermek için bir pull isteği gönderin.

    ```bash
    git commit -m "commit mesajı"
    git push origin master
    ```

3. Bir pull isteği göndermek için projenin Github sayfasına gidin ve "New Pull Request" düğmesine tıklayın.

# Özgünlük

Bu projede ilk başta veri seti ön işlemden geçmiştir. Genel ön işlemlerın dışında ayrıca etiket içerisindeki etiket uyuşmazlıkları giderilmiştir (Örneğin bir metnin offansive değeri 0 iken ırkçı söylem olarak etiketlenmesi gibi). Ardından iki farklı model geliştirildi. İlk olarak Türkçe metinler için genel bir aşağılayıcı söylem içeren yüksek başarımlı bir ikili sınıflandırma modeli geliştirildi. Burada veri seti farklı modellerden eğitilerek başarımı test edilmiştir. Bu modellerden Fine-Tuned BERT modeli testlerde en başarılı sonucu vermiştir. Geliştirilen modelden elde edilen test sonuçlarına göre F1 skoru yüzde 97.85 çıkmıştır. İkinci olarak aşağılayıcı söylemler için çok sınıflı bir sınıflandırma modeli geliştirilmiştir. Burada yine farklı modeller test edilmiş ve en başarılı sonucu yine Fine-Tuned BERT modeli vermiştir. Geliştirilen modelden elde edilen test sonuçlarına göre aşağılayıcı söylemi temsil eden dört etiketin Makro F1 skoru yüzde 95.07 çıkmıştır. Sonuç olarak geliştirilen çok etiketli sınıflandırma modeli ile Türkçe aşağılayıcı söylemler yüksek doğrulukta tespit edilebilmektedir.

# Lisans:

Bu proje GNU Affero General Public License v3 (AGPLv3) ile lisanslanmıştır. Lisans hakkında daha fazla bilgi için [LICENSE](https://github.com/hititddi/hititddiproject/blob/main/LICENSE) dosyasına bakınız.

# İletişim

Eğer herhangi bir sorunuz, öneriniz veya geri bildiriminiz varsa, lütfen proje sahibi/ekibiyle iletişime geçin.

* Emre DENİZ - emredeniz18@hotmail.com
* Harun Emre KIRAN - harunemrekiran@gmail.com

# Kaynaklar

* https://github.com/google-research/bert
* https://github.com/pytorch/pytorch
