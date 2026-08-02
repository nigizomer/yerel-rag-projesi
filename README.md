# Yerel RAG (Retrieval-Augmented Generation) Tabanlı Soru-Cevap Sistemi

Bu proje, staj çalışması kapsamında geliştirilmiş; tamamen yerel (çevrimdışı) ortamda çalışan, büyük dil modelleri (LLM) aracılığıyla uzun belgelerin (PDF) okunup analiz edilmesini sağlayan bir RAG mimarisidir.

## Proje Mimarisi ve Dosya Yapısı

Sistem, geliştirme sürecine paralel olarak 4 ana Python modülünden oluşmaktadır:

*   **`01_veritabani_olusturucu.py`**: Sistemin veri hazırlık aşamasıdır. Hedef PDF belgesini okur, metni 1000'er karakterlik bağlamsal parçalara böler (chunking) ve HuggingFace gömme (embedding) modeli ile vektörel formata dönüştürerek Chroma veritabanına kaydeder.
*   **`02_rag_asistani_hafif_model.py`**: Sistemin ilk prototipidir (Qwen 2.5 - 0.5B). Modelin boyutu küçük olduğu için donanımı yormasa da, karmaşık Türkçe dil yapısında ve geniş bağlamı kavramada yetersiz kalarak halüsinasyon (hallucination) eğilimi göstermiştir. Bu dosya, projenin gelişim sürecini ve optimizasyon adımlarını belgelemek adına korunmuştur.
*   **`03_rag_asistani_gelismis_model.py`**: Projenin nihai ve en verimli sürümüdür. İlk prototipteki halüsinasyon sorunlarını çözmek amacıyla 7 Milyar (7B) parametreli gelişmiş Qwen modeline geçiş yapılmıştır. Bu model, veri tabanından çekilen metinleri çok daha başarılı bir şekilde işleyerek Türkçe dilinde yüksek tutarlılıkla ve hatasız cevaplar üretmektedir.
*   **`04_pdf_ozetleyici_arac.py`**: Uzun belgelerin özetlenmesinde karşılaşılan hafıza (token) sınırını aşmak için sisteme entegre edilen özel modüldür. Belgeyi 5'er sayfalık döngüler halinde işleyerek sistemin tıkanmasını engeller ve bütüncül bir özet çıktısı sunar.
*   **`Yapay_Zeka_El_Kitabi_2026.pdf`**: Sistemin arama ve özetleme performansının test edildiği örnek veri kaynağıdır. (Sistem mimarisi, sayfa sınırı olmaksızın çok daha hacimli ve uzun belgelerde de sorunsuz çalışacak şekilde tasarlanmıştır).

## Karşılaşılan Zorluklar ve Çözümler

1.  **Halüsinasyon Sorunu ve Dil Desteği:**
    *   *Problem:* Başlangıçta kullanılan 0.5B parametreli model, belgedeki bilgileri sentezlemekte zorlanmış, sorulara belgede olmayan uydurma (halüsinatif) yanıtlar üretmiş ve Türkçe gramerinde yetersiz kalmıştır.
    *   *Çözüm:* Model mimarisi büyütülerek doğrudan Qwen 2.5 - 7B sürümüne entegre edildi. Bu donanımsal yükseltme sayesinde modelin Türkçe anlama yeteneği ve metne sadık kalma (grounding) oranı maksimize edildi.
2.  **Bağlam Penceresi (Token Limit) ve Ölçeklenebilirlik Kısıtlamaları:**
    *   *Problem:* Uzun hacimli belgelerin tamamının tek seferde modele özetletilmek istenmesi, modelin girdi sınırlarını (token limit) aşarak sistemin çökmesine veya eksik işlem yapmasına neden olmaktadır.
    *   *Çözüm:* `04_pdf_ozetleyici_arac.py` algoritması geliştirilerek, belgenin bütünü yerine belirli sayfa aralıklarıyla parça parça işlenmesi sağlandı. Bu modüler yaklaşım sayesinde sistem, sayfa sayısından tamamen bağımsız hale getirilerek her uzunluktaki belgeyi işleyebilir kapasiteye ulaştırıldı.

## Kullanılan Teknolojiler

*   **Büyük Dil Modeli (LLM):** Alibaba Qwen 2.5 (0.5B ve 7B)
*   **RAG Altyapısı:** LangChain
*   **Vektör Veritabanı:** Chroma DB
*   **Gömme (Embedding) Modeli:** HuggingFace (`all-MiniLM-L6-v2`)

##  Geliştirici Notu

Tarsus Üniversitesi Bilgisayar Mühendisliği bölümünde eğitimime devam ediyorum. Bu proje, staj sürecimde Büyük Dil Modelleri (LLM) ve Doğal Dil İşleme (NLP) mimarileri üzerinde teorik bilgilerimi pratiğe döktüğüm; veri işleme, hata ayıklama ve yerel yapay zeka sistemlerinin optimizasyonunu bizzat test ettiğim bir mühendislik çalışmasıdır.
