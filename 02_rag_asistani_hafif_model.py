from foundry_local_sdk import Configuration, FoundryLocalManager
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Modeli Başlatma
FoundryLocalManager.initialize(Configuration(app_name="my-app"))
model = FoundryLocalManager.instance.catalog.get_model("qwen2.5-0.5b")
model.load()
client = model.get_chat_client()

# 2. Kaydettiğimiz Veritabanını Yükle
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./rag_veritabani", embedding_function=embeddings)

def cevap_bul(soru):
    # Soruyu Veritabanında Ara
    ilgili_parcalar = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in ilgili_parcalar])
    

    sistem_mesaji = "Sen sadece sana verilen metne bakarak cevap veren bir asistansın. Metinde bilgi yoksa sadece 'BİLMİYORUM' yaz."
    
    kullanici_mesaji = f"METİN:\n{baglam}\n\nSORU: {soru}\nCEVAP:"
    
    # Modeli çalıştır
    response = client.complete_chat([
        {"role": "system", "content": sistem_mesaji},
        {"role": "user", "content": kullanici_mesaji}
    ])
    
    return response.choices[0].message.content

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 YENİ SİSTEM HAZIR! Qwen Asistanı Devrede.")
    print("   (Çıkmak için 'q' veya 'çıkış' yaz)")
    print("="*50)
    
    while True:
        kullanici_sorusu = input("\nSen: ")
        
        if kullanici_sorusu.lower() in ['q', 'çıkış', 'quit']:
            print("Kaptan köşkten ayrılıyor... Görüşmek üzere!")
            break
            
        print("Qwen düşünüyor (Veritabanı taranıyor)...")
        cevap = cevap_bul(kullanici_sorusu)
        print(f"\nQwen: {cevap}")
