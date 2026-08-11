from foundry_local_sdk import Configuration, FoundryLocalManager
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

# 1. Kendi Çalışan Model Kurulumumuz
FoundryLocalManager.initialize(Configuration(app_name="my-app"))
model_name = "qwen2.5-7b"
model = FoundryLocalManager.instance.catalog.get_model(model_name)
model.load() 

# Modelimizle konuşacağımız doğrudan istemci (client)
client = model.get_chat_client()

def main():
    # 2. PDF Dosyasını Bulma
    pdf_path = Path("bolum-100/Yapay_Zeka_El_Kitabi_2026.pdf")

    if not pdf_path.exists():
        print(f"HATA: Klasörde '{pdf_path}' dosyası bulunamadı!")
        return

    # 3. PDF'i Yükleme ve Sayfalama
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"Harika! PDF başarıyla yüklendi. Toplam {len(documents)} sayfa işlenecek.\n")

    # 4. Token Limitine Takılmamak İçin Sayfaları 5'er 5'er Okutma
    chunk_size = 5
    summaries = []

    for i in range(0, len(documents), chunk_size):
        chunk = documents[i : i + chunk_size]
        chunk_content = "\n".join([doc.page_content for doc in chunk])
        
        ilerleme_ilk = i + 1
        ilerleme_son = min(i + chunk_size, len(documents))
        print(f"Yapay zeka {ilerleme_ilk} ile {ilerleme_son}. sayfalar arasını özetliyor, lütfen bekleyin...")

        # 5. Kendi Sohbet Kodumuzla Özetletme
        sistem_mesaji = "Sen belgeleri inceleyen ve özetleyen yardımcı bir asistansın. Lütfen sana verilen metni Türkçe olarak özetle."
        kullanici_mesaji = f"Lütfen şu metni özetle:\n\n{chunk_content}"

        response = client.complete_chat([
            {"role": "system", "content": sistem_mesaji},
            {"role": "user", "content": kullanici_mesaji}
        ])
        
        # Gelen cevabı listemize ekliyoruz
        summaries.append(response.choices[0].message.content)

    # 6. Tüm Özetleri Ekrana Yazdırma
    print("\n" + "="*60)
    print("=== YAPAY ZEKA EL KİTABI GENEL ÖZETİ ===")
    for idx, ozet in enumerate(summaries):
        print(f"\n--- Kısım {idx+1} Özeti ---")
        print(ozet)
    print("="*60)

if __name__ == "__main__":
    main()
